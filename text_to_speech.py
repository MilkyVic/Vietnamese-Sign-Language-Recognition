import argparse
import os
from pathlib import Path

from openai import OpenAI

from load_env import load_dotenv


def synthesize_speech(text: str, output_path: Path, voice: str = "coral") -> Path:
    if not text:
        raise ValueError("Text must not be empty.")

    client = OpenAI()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    format_ext = output_path.suffix.lstrip(".") or "mp3"
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        response_format=format_ext,
        instructions="Hãy nói theo tone giọng tiếng Việt, nhấn nhá đầy đủ, lên xuống tự nhiên như giọng miền Trung Việt Nam.",
        speed=1.25,
    ) as response:
        response.stream_to_file(output_path)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert recognized text to speech using OpenAI gpt-4o-mini-tts.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Raw text to synthesize.")
    group.add_argument("--input-file", type=Path, help="Path to a UTF-8 text file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Outputs/audio_output.mp3"),
        help="Path to save synthesized audio (extension determines format, default mp3).",
    )
    parser.add_argument("--voice", type=str, default="alloy", help="Voice preset supported by gpt-4o-mini-tts.")
    return parser.parse_args()


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    args = parse_args()
    text = args.text
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

    output_path = synthesize_speech(text, args.output, args.voice)
    print(f"Synthesized audio saved to {output_path}")


if __name__ == "__main__":
    main()
