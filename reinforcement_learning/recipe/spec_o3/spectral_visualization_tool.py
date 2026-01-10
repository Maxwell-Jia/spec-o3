# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import logging
import os
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import matplotlib
import ray

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from recipe.astro.utils import plot_spectrum
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class VisualExecutionWorker:
    """Worker for executing visual processing operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing visual processing: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_visual_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize visual execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(VisualExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class SpectralVisualizationTool(BaseTool):
    """A tool for visualizing spectral data with wavelength range zooming capabilities.

    This tool provides spectral visualization with support for focusing on specific wavelength
    ranges, enabling AI models to analyze detailed spectral features for accurate classification.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the spectral visualization operation
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 30)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_visual_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )
        logger.info(f"Initialized SpectralVisualizationTool with config: {config}")

    def _validate_wavelength_range(self, wavelength_range: list, full_wavelength_range: tuple) -> bool:
        """Validate the wavelength range parameters."""
        try:
            if len(wavelength_range) != 2:
                logger.warning(f"wavelength_range must have exactly 2 elements, got {len(wavelength_range)}")
                return False

            min_wave, max_wave = wavelength_range
            if not (min_wave < max_wave):
                logger.warning(f"Invalid wavelength range: min={min_wave}, max={max_wave}")
                return False

            # Check if range is within the available data range
            data_min, data_max = full_wavelength_range
            if max_wave < data_min or min_wave > data_max:
                logger.warning(
                    f"Wavelength range [{min_wave}, {max_wave}] is outside data range [{data_min}, {data_max}]"
                )
                return False

            return True
        except Exception as e:
            logger.warning(f"Wavelength range validation error: {e}")
            return False

    def _convert_matplotlib_to_image(self, fig):
        """Convert matplotlib figure to PIL Image."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        buf.seek(0)

        # Convert to PIL Image and copy data to avoid closed buffer issues
        from PIL import Image

        img = Image.open(buf)
        # Create a copy to avoid issues with closed buffer
        img_copy = img.copy()
        buf.close()
        plt.close(fig)  # Important: close the figure to free memory

        return img_copy

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Creates a new instance for spectral visualization tool.

        This method initializes a new session for spectral data, which can then be used
        for visualization operations like wavelength range focusing.

        Args:
            instance_id: An optional unique identifier for the instance. If not
                provided, a new UUID will be generated.
            **kwargs: Should contain spectral data including:
                - 'wavelength': array-like wavelength data
                - 'flux': array-like flux data
                - 'redshift': redshift value (optional)
                - 'title': plot title (optional)

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Get required spectral data from kwargs
        wavelength = kwargs.get("wavelength")
        flux = kwargs.get("flux")
        if wavelength is None or flux is None:
            raise ValueError("Missing required 'wavelength' and 'flux' parameters in kwargs")

        # Convert to numpy arrays for processing
        wavelength = np.array(wavelength)
        flux = np.array(flux)

        if len(wavelength) != len(flux):
            raise ValueError("wavelength and flux must have the same length")

        # Store instance data
        self._instance_dict[instance_id] = {
            "wavelength": wavelength,
            "flux": flux,
            "redshift": kwargs.get("redshift", None),
            "title": kwargs.get("title", "Spectrum"),
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the spectral visualization with optional wavelength range focusing."""
        wavelength_range = parameters.get("wavelength_range")
        label = parameters.get("label", "")

        instance_data = self._instance_dict[instance_id]
        wavelength = instance_data["wavelength"]
        flux = instance_data["flux"]
        redshift = instance_data["redshift"]
        title = instance_data["title"]

        try:
            # Apply redshift correction to get rest frame wavelength for validation
            if redshift:
                rest_wavelength = wavelength / (1 + redshift)
            else:
                rest_wavelength = wavelength

            # Validate wavelength range if provided
            if wavelength_range is not None:
                full_range = (rest_wavelength.min(), rest_wavelength.max())
                if not self._validate_wavelength_range(wavelength_range, full_range):
                    error_msg = (
                        f"Error: The specified wavelength range {wavelength_range} is invalid or "
                        f"outside the available data range [{full_range[0]:.1f}, {full_range[1]:.1f}]."
                    )
                    logger.warning(f"Tool execution failed: {error_msg}")
                    return ToolResponse(text=error_msg), -0.05, {"success": False}

                # Convert to tuple for plot_spectrum function
                wavelength_range = tuple(wavelength_range)

            # Generate the spectral plot using the utils function
            fig, _ = plot_spectrum(
                wavelength=rest_wavelength, flux=flux, wavelength_range=wavelength_range, title=title
            )

            # Convert matplotlib figure to PIL Image
            spectrum_image = self._convert_matplotlib_to_image(fig)
            logger.info(f"Generated spectral plot with size: {spectrum_image.size}")

        except Exception as e:
            logger.error(f"Error processing spectral visualization: {e}")
            return ToolResponse(text=f"Error processing spectral visualization: {e}"), -0.05, {"success": False}

        # Generate response text
        if wavelength_range is not None:
            response_text = f"Generated spectral visualization focused on wavelength range {wavelength_range} Å."
            if label:
                response_text = (
                    f"Generated spectral visualization focused on wavelength range {wavelength_range} Å for {label}."
                )
        else:
            response_text = "Generated full spectrum visualization."
            if label:
                response_text = f"Generated full spectrum visualization for {label}."

        return (
            ToolResponse(
                image=[spectrum_image],
                text=response_text,
            ),
            0.0,
            {"success": True},
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance and clean up resources."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


if __name__ == "__main__":
    import asyncio

    from datasets import load_dataset

    from verl.tools.schemas import OpenAIFunctionParametersSchema, OpenAIFunctionPropertySchema, OpenAIFunctionSchema

    async def test_spectral_tool():
        # Load test data
        dataset = load_dataset(
            "parquet", data_files=["/data/group/project3/agentic_rl/tmp_data/tns_sn_classification_train.parquet"]
        )
        df = dataset["train"].to_pandas()

        # Get first sample data for testing
        wavelength = df.iloc[0]["wavelength"]
        flux = df.iloc[0]["flux"]
        redshift = df.iloc[0]["redshift"]

        # Create correct tool schema
        tool_schema = OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="spectral_visualization",
                description="A tool for visualizing spectral data with wavelength range zooming capabilities.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "wavelength_range": OpenAIFunctionPropertySchema(
                            type="array",
                            description="Wavelength range to focus on [min_wavelength, max_wavelength] in Angstroms",
                        ),
                        "label": OpenAIFunctionPropertySchema(
                            type="string", description="Optional label for the spectral feature being analyzed"
                        ),
                    },
                    required=[],
                ),
            ),
        )

        # Initialize tool
        tool = SpectralVisualizationTool(config={}, tool_schema=tool_schema)

        # Test 1: Create instance with spectral data
        print("=== Test 1: Creating tool instance ===")
        instance_id, create_response = await tool.create(
            wavelength=wavelength, flux=flux, redshift=redshift, title="Test Supernova Spectrum"
        )
        print(f"Instance ID: {instance_id}")
        print(f"Create response: {create_response}")

        # Test 2: Full spectrum visualization
        print("\n=== Test 2: Full spectrum visualization ===")
        parameters = {"label": "Full spectrum"}
        response, reward, success = await tool.execute(instance_id, parameters)
        print(f"Response text: {response.text}")
        print(f"Reward: {reward}")
        print(f"Success: {success}")

        # Save full spectrum image
        if response.image and len(response.image) > 0:
            response.image[0].save("/data/group/project3/agentic_rl/verl/recipe/astro/test_full_spectrum.png")
            print("Full spectrum saved as test_full_spectrum.png")

        # Test 3: Wavelength range focused visualization (H-alpha region around 6563Å)
        print("\n=== Test 3: H-alpha region visualization ===")
        parameters = {"wavelength_range": [6500, 6600], "label": "H-alpha region"}
        response, reward, success = await tool.execute(instance_id, parameters)
        print(f"Response text: {response.text}")
        print(f"Reward: {reward}")
        print(f"Success: {success}")

        # Save wavelength range image
        if response.image and len(response.image) > 0:
            response.image[0].save("/data/group/project3/agentic_rl/verl/recipe/astro/test_halpha_region.png")
            print("H-alpha region saved as test_halpha_region.png")

        # Test 4: Try invalid wavelength range
        print("\n=== Test 4: Invalid wavelength range ===")
        parameters = {"wavelength_range": [10000, 12000], "label": "Invalid range"}
        response, reward, success = await tool.execute(instance_id, parameters)
        print(f"Response text: {response.text}")
        print(f"Reward: {reward}")
        print(f"Success: {success}")

        # Clean up
        await tool.release(instance_id)
        print("\n=== Test completed ===")

    # Run async test
    asyncio.run(test_spectral_tool())
