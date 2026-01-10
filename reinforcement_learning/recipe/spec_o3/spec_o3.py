# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import re

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def _match_supernova_class(text: str, valid_classes: list) -> str:
    """
    Flexible matching for supernova class names.

    Args:
        text: Text to match against valid classes
        valid_classes: List of valid supernova class names

    Returns:
        Matched class name or empty string if no match
    """
    text_clean = text.strip()

    # First try exact match (case insensitive)
    for cls in valid_classes:
        if text_clean.upper() == cls.upper():
            return cls

    # Try partial matching for common variations
    text_upper = text_clean.upper()
    for cls in valid_classes:
        cls_upper = cls.upper()

        # Handle variations like "SN IA" -> "SN Ia (Normal)"
        if cls_upper == "SN IA (NORMAL)" and text_upper in ["SN IA", "SN Ia", "IA"]:
            return cls
        elif cls_upper == "SN IAX[02CX-LIKE]" and text_upper in ["SN IAX", "SN Iax", "IAX"]:
            return cls
        elif cls_upper.replace(" ", "").replace("-", "").replace("[", "").replace("]", "") == text_upper.replace(
            " ", ""
        ).replace("-", "").replace("[", "").replace("]", ""):
            return cls

    return ""


def _extract_answer_fallback(text: str, valid_classes: list, task_type: str) -> str:
    """
    Fallback method to extract answer from text when boxed format is missing.

    Args:
        text: Text to search for answer
        valid_classes: List of valid classes
        task_type: "binary" or "multiclass"

    Returns:
        Extracted answer or empty string if no match
    """
    if task_type == "binary":
        # For binary classification, use regex patterns
        patterns = [r"\b(YES)\b", r"\b(NO)\b"]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
    else:
        # For multiclass, try to find any valid class in the text
        for cls in valid_classes:
            # Try exact match first
            if cls.upper() in text.upper():
                return cls

            # Try flexible matching
            matched = _match_supernova_class(text, [cls])
            if matched:
                return matched

    return ""


class CustomRLHFDataset(RLHFDataset):
    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict and row_dict.get(self.image_key, None) is not None:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205  # noqa: E501
                multi_modal_data["image"] = images

            model_inputs = self.processor(text=[raw_prompt], images=images, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        # tools_kwargs = {
        #     "image_zoom_in_tool": {
        #         "create_kwargs": {"image": images[0]},
        #         # "execute_kwargs": {},
        #         # "calc_reward_kwargs": {},
        #         # "release_kwargs": {},
        #     }
        # }
        row_dict["index"] = index
        # row_dict["tools_kwargs"] = tools_kwargs
        row_dict["agent_name"] = "tool_agent"
        return row_dict


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None, **kwargs) -> dict:
    """
    Compute reward score for astro model solutions.

    Returns a dictionary containing:
    - 'score': Final weighted reward score for GRPO training
    - Additional classification metrics for validation purposes

    Reward components:
    - Accuracy reward (1.0 weight): Whether the answer matches the ground truth
    - Format reward (0.2 weight): Whether the output follows expected \\boxed{} format
    - Tool reward (1.0 weight): Whether tools were used when answer is correct

    Args:
        data_source: Source of the data (lamost_cv for YES/NO, tns_sn for 10-class)
        solution_str: Model's solution string
        ground_truth: Ground truth answer
        extra_info: Dictionary to store additional metrics for validation (legacy parameter)

    Returns:
        Dictionary with 'score' field and additional classification metrics
    """

    # Define valid answer classes based on data source
    if data_source in ["cv", "carbon", "cc", "ss", "wd", "gm"]:
        # YES/NO binary classification
        valid_classes = ["YES", "NO"]
        task_type = "binary"
    elif data_source == "lamost_cv_galaxy":
        # 3-class classification: CV / GALAXY / OTHER
        valid_classes = ["CV", "GALAXY", "OTHER"]
        task_type = "multiclass"
    elif data_source == "tns":
        # 10-class supernova classification
        valid_classes = [
            "SN IIP",
            "SN IIb",
            "SN Ia (Normal)",
            "SN Ia-91T-like",
            "SN Ia-91bg-like",
            "SN Iax[02cx-like]",
            "SN Ib",
            "SN Ibn",
            "SN Ic",
            "SN Ic-BL",
        ]
        task_type = "multiclass"
    else:
        # Default to binary classification for backward compatibility
        valid_classes = ["YES", "NO"]
        task_type = "binary"
        logger.warning(f"Unknown data_source: {data_source}, defaulting to binary classification")

    # Initialize tracking variables
    is_format_error = False

    # 1. Strict format validation for structured response
    answer_text = ""

    # Check for required structure: <think>...</think> and <answer>...</answer>
    # Check think tags - allow multiple pairs but must be properly matched
    think_open_count = len(re.findall(r"<think>", solution_str))
    think_close_count = len(re.findall(r"</think>", solution_str))

    # Check answer tags - must be exactly one pair
    answer_open_count = len(re.findall(r"<answer>", solution_str))
    answer_close_count = len(re.findall(r"</answer>", solution_str))
    answer_matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)

    # Validate think tags - allow multiple but must be properly paired
    if think_open_count != think_close_count or think_open_count == 0:
        is_format_error = True
        if think_open_count != think_close_count:
            logger.debug(f"Mismatched <think> tags: {think_open_count} opening, {think_close_count} closing")
        else:
            logger.debug("Missing <think>...</think> tags")

    # Validate answer tags - must be exactly one pair
    if answer_open_count != 1 or answer_close_count != 1:
        is_format_error = True
        logger.debug(
            f"Invalid <answer> tags: {answer_open_count} opening, {answer_close_count} closing - must be exactly 1 pair"
        )

    if not answer_matches:
        is_format_error = True
        logger.debug("Missing <answer>...</answer> tags")
    elif len(answer_matches) != 1:
        # Multiple <answer> tags found - this is a format error since we only allow exactly one pair
        is_format_error = True
        logger.debug(f"Multiple <answer> tags found: {len(answer_matches)} - only exactly 1 pair allowed")
        # Use the last answer tag as fallback
        answer_content = answer_matches[-1].strip()

        # Look for \\boxed{...} pattern within the answer section
        boxed_match = re.search(r"\\boxed\{([^}]*)\}", answer_content)
        if boxed_match:
            boxed_content = boxed_match.group(1).strip()

            # Validate based on task type
            if task_type == "binary":
                if boxed_content.upper() in valid_classes:
                    answer_text = boxed_content.upper()
                else:
                    is_format_error = True
                    logger.debug(f"Invalid boxed content: '{boxed_content}', expected {valid_classes}")
                    answer_text = boxed_content  # Use as-is for partial credit
            else:  # multiclass
                # For multiclass, use flexible matching
                matched_class = _match_supernova_class(boxed_content, valid_classes)
                if matched_class:
                    answer_text = matched_class
                else:
                    is_format_error = True
                    logger.debug(f"Invalid boxed content: '{boxed_content}', expected one of {valid_classes}")
                    answer_text = boxed_content  # Use as-is for partial credit
        else:
            # No \\boxed{} found within <answer> tags - this is a format error
            is_format_error = True
            logger.debug("Missing \\boxed{} within <answer> tags")

            # Fallback: try to extract valid class from answer content
            answer_text = _extract_answer_fallback(answer_content, valid_classes, task_type)
    else:
        # Exactly one <answer> tag found - normal case
        answer_content = answer_matches[0].strip()

        # Look for \\boxed{...} pattern within the answer section
        boxed_match = re.search(r"\\boxed\{([^}]*)\}", answer_content)
        if boxed_match:
            boxed_content = boxed_match.group(1).strip()

            # Validate based on task type
            if task_type == "binary":
                if boxed_content.upper() in valid_classes:
                    answer_text = boxed_content.upper()
                else:
                    is_format_error = True
                    logger.debug(f"Invalid boxed content: '{boxed_content}', expected {valid_classes}")
                    answer_text = boxed_content  # Use as-is for partial credit
            else:  # multiclass
                # For multiclass, use flexible matching
                matched_class = _match_supernova_class(boxed_content, valid_classes)
                if matched_class:
                    answer_text = matched_class
                else:
                    is_format_error = True
                    logger.debug(f"Invalid boxed content: '{boxed_content}', expected one of {valid_classes}")
                    answer_text = boxed_content  # Use as-is for partial credit
        else:
            # No \\boxed{} found within <answer> tags - this is a format error
            is_format_error = True
            logger.debug("Missing \\boxed{} within <answer> tags")

            # Fallback: try to extract valid class from answer content
            answer_text = _extract_answer_fallback(answer_content, valid_classes, task_type)

    # Final fallback if no structured format found
    if not answer_text and is_format_error:
        # First try to extract \boxed{} from entire solution
        boxed_match = re.search(r"\\boxed\{([^}]*)\}", solution_str)
        if boxed_match:
            boxed_content = boxed_match.group(1).strip()
            if task_type == "binary":
                if boxed_content.upper() in valid_classes:
                    answer_text = boxed_content.upper()
                else:
                    answer_text = boxed_content  # Use as-is for partial credit
            else:  # multiclass
                matched_class = _match_supernova_class(boxed_content, valid_classes)
                if matched_class:
                    answer_text = matched_class
                else:
                    answer_text = boxed_content  # Use as-is for partial credit

        # If no \boxed{} found, try to extract valid class from entire solution
        if not answer_text:
            answer_text = _extract_answer_fallback(solution_str, valid_classes, task_type)

        # If still no match, use the last line as fallback
        if not answer_text:
            lines = [line.strip() for line in solution_str.split("\n") if line.strip()]
            if lines:
                answer_text = lines[-1]

    # Clean up answer text
    answer_text = answer_text.strip()

    # If answer is still empty, mark as format error and use full solution
    if not answer_text:
        is_format_error = True
        answer_text = solution_str.strip()

    # 2. Compare with ground truth using flexible matching
    acc_reward = 0.0

    # ULTRA STRICT: No normalization, use raw text with minimal preprocessing
    # Only basic whitespace cleanup to avoid trivial formatting differences
    def basic_cleanup(text):
        """Basic text cleanup without semantic normalization"""
        return re.sub(r"\s+", " ", text).strip()

    cleaned_answer = basic_cleanup(answer_text)
    cleaned_ground_truth = basic_cleanup(ground_truth)

    # ASYMMETRIC REWARD: Differentiate between TP/TN/FP/FN to combat class imbalance
    # This addresses reward hacking where model always predicts majority class (NO)
    #
    # Design rationale for 1:10 positive:negative ratio:
    # - TP (True Positive): Highest reward - finding rare positive samples is valuable
    # - TN (True Negative): Small reward - correct but common outcome
    # - FN (False Negative): Severe penalty - missing a rare positive is costly
    # - FP (False Positive): Moderate penalty - false alarm is less severe than missing target
    #
    # Expected reward analysis:
    # - Always predict NO: 0.1*(-1.5) + 0.9*(0.2) = -0.15 + 0.18 = +0.03 (slightly positive)
    # - Always predict YES: 0.1*(1.5) + 0.9*(-0.3) = 0.15 - 0.27 = -0.12 (negative)
    # - Perfect classifier: 0.1*(1.5) + 0.9*(0.2) = 0.15 + 0.18 = +0.33 (best)
    #
    # The design encourages learning to find positives while maintaining reasonable TN rate

    is_correct = cleaned_answer == cleaned_ground_truth
    is_positive_sample = (cleaned_ground_truth == "YES") if task_type == "binary" else (cleaned_ground_truth != "NO")

    if is_correct and is_positive_sample:
        # True Positive: Correctly identified positive sample
        acc_reward = 1.0
    elif is_correct and not is_positive_sample:
        # True Negative: Correctly identified negative sample
        acc_reward = 1.0
    elif not is_correct and is_positive_sample:
        # False Negative: Missed a positive sample (most severe error)
        acc_reward = 0.0
    elif not is_correct and not is_positive_sample:
        # False Positive: False alarm on negative sample
        acc_reward = 0.0

    # Note: For multiclass tasks (task_type != "binary"), this uses a simplified binary view
    # where any non-default class is considered "positive". This can be refined further if needed.

    # Penalize excessively long answers
    # if len(answer_text) >= 500:
    #     acc_reward = 0.0
    #     is_format_error = True

    # 3. Check tool usage - look for tool_call/tool_response patterns
    has_tool_usage = bool(
        re.search(r"<tool_call>.*?</tool_call>", solution_str, re.DOTALL)
        # and re.search(r"<tool_response>.*?</tool_response>", solution_str, re.DOTALL)
        # and "Generated spectral visualization" in solution_str
    )

    # Tool reward: only give if tools were used AND answer is correct
    # tool_reward = 1.0 if has_tool_usage and acc_reward > 0.5 else 0.0
    tool_reward = 1.0 if has_tool_usage else 0.0

    # Format reward: penalty for format errors (no \\boxed{})
    format_reward = -1.0 if is_format_error else 0.0

    # Log debug information for problematic cases
    if is_format_error or acc_reward == 0.0:
        logger.debug(
            f"Scoring details:\n"
            f"Solution: {solution_str[:200]}...\n"
            f"Extracted answer: '{answer_text}'\n"
            f"Cleaned answer: '{cleaned_answer}'\n"
            f"Cleaned ground truth: '{cleaned_ground_truth}'\n"
            f"Format error: {is_format_error}\n"
            f"Tool usage: {has_tool_usage}\n"
            f"Accuracy reward: {acc_reward}\n"
            f"Format reward: {format_reward}\n"
            f"Tool reward: {tool_reward}"
        )

    # Store additional information for validation metrics
    if extra_info is not None:
        # Binary accuracy (1.0 if correct, 0.0 if incorrect)
        extra_info["acc"] = 1.0 if acc_reward > 0.5 else 0.0

        # Store predictions and ground truth for classification metrics
        extra_info["pred"] = cleaned_answer
        extra_info["gt"] = cleaned_ground_truth

        # Additional metrics
        extra_info["format_correct"] = 0.0 if is_format_error else 1.0
        extra_info["tool_used"] = 1.0 if has_tool_usage else 0.0
        extra_info["raw_answer"] = answer_text
        extra_info["raw_solution"] = solution_str[:200]  # Truncated for logging

    # Final weighted score
    # final_score = 1.0 * acc_reward + 0.2 * format_reward + 1.0 * tool_reward
    final_score = 1.0 * acc_reward + 0.2 * format_reward
    
    # Prepare result dictionary with score and additional metrics
    result = {
        "score": final_score,  # Primary reward score for GRPO training
        "predicted_class": cleaned_answer,
        "true_class": cleaned_ground_truth,
    }

    return result
