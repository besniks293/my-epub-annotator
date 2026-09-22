import argparse
import os
import re
import sys
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class WordAnnotation(BaseModel):
    word: str = Field(description="The exact word identified as archaic, obscure, or uncommon.")
    definition: str = Field(description="A concise footnote explanation (10-25 words) suited for contemporary readers.")


class ChapterAnnotations(BaseModel):
    annotations: list[WordAnnotation] = Field(default_factory=list)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate EPUB with LLM-generated footnotes for obscure/archaic words."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input EPUB file")
    parser.add_argument("--output", "-o", required=True, help="Path to output EPUB file")
    return parser.parse_args()


def process_text_chunk(client: genai.Client, text_content: str) -> list[WordAnnotation]:
    if not text_content.strip():
        return []

    prompt = f"""
    Analyze the following text from a book chapter. Identify words or short phrases that a 
    contemporary general reader would likely not understand (archaic, rare, highly domain-specific, 
    or obsolete vocabulary). 

    For each identified word, provide a brief, clear footnote explanation based on its context.

    Text:
    {text_content}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ChapterAnnotations,
                temperature=0.2,
            ),
        )
        result = ChapterAnnotations.model_validate_json(response.text)
        return result.annotations
    except Exception as e:
        print(f"Error during API call: {e}", file=sys.stderr)
        return []


def annotate_html(soup: BeautifulSoup, annotations: list[WordAnnotation]) -> None:
    if not annotations:
        return

    footnote_section = soup.find("section", class_="footnotes")
    if not footnote_section:
        footnote_section = soup.new_tag("section", attrs={"class": "footnotes", "epub:type": "footnotes"})
        footnote_hr = soup.new_tag("hr")
        footnote_list = soup.new_tag("ol")
        footnote_section.append(footnote_hr)
        footnote_section.append(footnote_list)
        if soup.body:
            soup.body.append(footnote_section)
        else:
            soup.append(footnote_section)
    else:
        footnote_list = footnote_section.find("ol")
        if not footnote_list:
            footnote_list = soup.new_tag("ol")
            footnote_section.append(footnote_list)

    existing_count = len(footnote_list.find_all("li"))

    for idx, item in enumerate(annotations, start=existing_count + 1):
        target_word = item.word.strip()
        if not target_word:
            continue

        fn_id = f"fn{idx}"
        ref_id = f"fnref{idx}"

        # Match exact word boundaries (case-insensitive)
        pattern = re.compile(rf"\b({re.escape(target_word)})\b", re.IGNORECASE)

        replaced = False
        for text_node in soup.find_all(text=True):
            if text_node.parent.name in ["script", "style", "sup", "a"]:
                continue

            match = pattern.search(text_node)
            if match:
                matched_str = match.group(1)
                before = text_node[: match.start()]
                after = text_node[match.end() :]

                span = soup.new_tag("span", attrs={"class": "annotated-word"})
                span.string = matched_str

                sup = soup.new_tag("sup")
                a_ref = soup.new_tag("a", href=f"#{fn_id}", id=ref_id, attrs={"epub:type": "noteref"})
                a_ref.string = str(idx)
                sup.append(a_ref)

                new_elements = [soup.new_string(before), span, sup, soup.new_string(after)]
                text_node.replace_with(*new_elements)
                replaced = True
                break

        if replaced:
            li = soup.new_tag("li", id=fn_id, attrs={"epub:type": "footnote"})
            p = soup.new_tag("p")
            p.string = f"{item.word}: {item.definition} "

            a_back = soup.new_tag("a", href=f"#{ref_id}", attrs={"epub:type": "backlink"})
            a_back.string = "↩"
            p.append(a_back)
            li.append(p)
            footnote_list.append(li)


def process_epub(input_path: str, output_path: str) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)
    book = epub.read_epub(input_path)

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_content().decode("utf-8")
        soup = BeautifulSoup(content, "html.parser")

        text_blocks = [tag.get_text() for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "li"])]
        full_text = "\n".join(text_blocks)

        if len(full_text.strip()) < 100:
            continue

        print(f"Processing item: {item.get_name()}...")
        annotations = process_text_chunk(client, full_text)
        if annotations:
            annotate_html(soup, annotations)
            item.set_content(str(soup).encode("utf-8"))

    epub.write_epub(output_path, book)
    print(f"Annotated EPUB saved to: {output_path}")


if __name__ == "__main__":
    args = parse_arguments()
    process_epub(args.input, args.output)
