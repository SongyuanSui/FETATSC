import openai
import time
import json
import io
from transformers import AutoTokenizer
import requests

class BaseLLM:
    def __init__(self, model_name: str, api_key: str, base_url: str = None):
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def _make_api_call(self, messages: list, options: dict) -> list:
        start_time = time.time()
        responses = None
        retry_num = 0
        retry_limit = 2
        error = None

        print(f"Making API call to model: {self.model_name}")

        while responses is None:
            try:
                api_call_start = time.time()
                responses = self.client.chat.completions.create(
                    model=self.model_name,
                    seed=42,
                    messages=messages,
                    **options
                )
                api_call_time = time.time() - api_call_start
                print(f"API call completed in {api_call_time:.2f} seconds")
                error = None
            except openai.OpenAIError as e:
                print(f"OpenAI API Error: {e}", flush=True)
                error = str(e)
                if "This model's maximum context length is" in str(e):
                    print("Warning: Input exceeds max context length. Returning placeholder response.")
                    responses = {"choices": [{"message": {"content": "PLACEHOLDER"}}]}
                elif retry_num >= retry_limit:
                    print("Too many retry attempts. Returning placeholder response.")
                    responses = {"choices": [{"message": {"content": "PLACEHOLDER"}}]}
                else:
                    print(f"Retrying in 10 seconds... (attempt {retry_num + 1}/{retry_limit + 1})")
                    time.sleep(10)
                retry_num += 1

        total_time = time.time() - start_time
        print(f"Total API call process completed in {total_time:.2f} seconds")

        if error:
            raise Exception(error)

        results = [(res.message.content, None) for res in responses.choices]
        return results

    def generate_response(self, prompt: str, system_content: str = "You are an expert in time series data.") -> list:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        options = self.get_options()
        return self._make_api_call(messages, options)

    def generate(self, prompt: str) -> str:
        response = self.generate_response(prompt)
        return response[0][0]

    def get_options(self) -> dict:
        raise NotImplementedError

class MyDeepSeek(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(
            model_name=model_name,
            api_key="API-KEY",
            base_url="BASE-URL"
        )

    def get_options(self) -> dict:
        return {
            "temperature": 0.0,
            "top_p": 1.0,
        }


class MyChatGPT(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(
            model_name=model_name,
            api_key="API-KEY",
        )

    def get_options(
        self,
        temperature: float = 1.0,
        per_example_max_decode_steps: int = 300,
        per_example_top_p: float = 1.0,
        n_sample: int = 1,
        prompt: str = None,
    ) -> dict:
        return {
            "temperature": temperature,
            "n": n_sample,
            "top_p": per_example_top_p,
        }

class MyQwen(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(
            model_name=model_name,
            api_key="API-KEY",
            base_url="API-KEY"
        )

    def get_options(self) -> dict:
        return {
            "temperature": 0.0,
            "top_p": 1.0,
        }


class MyLlaMa3:
    def __init__(self, model_name: str):
        endpoint_url = "ENDPOINT-URL"
        self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        self.hf_token = "HF-TOKEN"
        self.endpoint_url = endpoint_url.rstrip("/")

        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=self.hf_token, use_fast=True
        )

        self._is_llama3 = "llama-3" in self.model_name.lower()

    def _build_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            sys_part = f"{system_prompt}" if system_prompt else ""
            return (
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n"
                f"{sys_part}\n<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"{user_prompt}\n<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n"
            )

    def get_model_options(
        self,
        temperature: float = 0.01,
        per_example_max_decode_steps: int = 3000,
        per_example_top_p: float = 0.99,
        n_sample: int = 1,
        prompt: str = None,
    ) -> dict:

        return {
            "temperature": temperature,
            "n": n_sample,
            "top_p": per_example_top_p,
            "do_sample": temperature > 0.01,
            "return_full_text": False,
        }

    def _extract_generated_text(self, hf_output) -> str:
        if isinstance(hf_output, dict) and "generated_text" in hf_output:
            return hf_output["generated_text"]
        if isinstance(hf_output, list) and hf_output and isinstance(hf_output[0], dict):
            if "generated_text" in hf_output[0]:
                return hf_output[0]["generated_text"]
            # Some runtimes return [{"generated_text": "..."}] or [{"summary_text": "..."}]
            if "summary_text" in hf_output[0]:
                return hf_output[0]["summary_text"]
        # If schema differs (e.g., error)
        return "PLACEHOLDER"

    def _core_generate(self, full_prompt: str, options: dict = None) -> str:
        if options is None:
            options = self.get_model_options(prompt=full_prompt)

        params = {
            "temperature": options["temperature"],
            "top_p": options["top_p"],
            "do_sample": options["do_sample"],
            "return_full_text": options["return_full_text"],
        }
        if self._is_llama3:
            params["stop"] = ["<|eot_id|>"]

        payload = {"inputs": full_prompt, "parameters": params}

        response = requests.post(self.endpoint_url, headers=self.headers, json=payload, timeout=60)
        if response.status_code != 200:
            raise RuntimeError(f"Inference error {response.status_code}: {response.text}")

        hf_output = response.json()
        generated_text = self._extract_generated_text(hf_output)
        return generated_text

    def generate(self, prompt: str, options: dict = None, returnall: bool = False):
        start_time = time.time()
        system_prompt = (
            "You are an expert in time series data."
        )
        full_prompt = self._build_chat_prompt(system_prompt, prompt)

        if options is None:
            options = self.get_model_options(prompt=full_prompt)

        n_sample = options.get("n", 1)
        results = []
        for _ in range(n_sample):
            response = self._core_generate(full_prompt, options)
            results.append(response)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{self.model_name} Generate Method elapsed time: {elapsed_time:.2f} seconds")
        return results if returnall else results[0]