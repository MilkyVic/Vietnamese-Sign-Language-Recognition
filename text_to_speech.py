import argparse
import os
from pathlib import Path

from openai import OpenAI

from load_env import load_dotenv


def synthesize_speech(
    text: str,
    output_path: Path,
    voice: str | None = None,
    instruction: str | None = None,
) -> Path:
    """
    Convert text to speech using OpenAI TTS.
    Reads only the provided output text (no prepended instruction by default).
    """

    if not text:
        raise ValueError("Text must not be empty.")

    client = OpenAI()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer explicit voice, else env TTS_VOICE, else default to "coral" (Vietnamese-friendly)
    selected_voice = voice or os.getenv("TTS_VOICE", "coral")

    # Only read the output text; no instruction prefix unless explicitly provided.
    final_text = text if instruction is None else f"{instruction} {text}".strip()

    format_ext = output_path.suffix.lstrip(".") or "mp3"
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=selected_voice,
        input=final_text,
        response_format=format_ext,
        speed=0.9,
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
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Optional instruction prefix (style/voice guidance). If omitted, uses env TTS_INSTRUCTION.",
    )
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

    output_path = synthesize_speech(text, args.output, args.voice, instruction=args.instruction)
    print(f"Synthesized audio saved to {output_path}")


if __name__ == "__main__":
    main()
