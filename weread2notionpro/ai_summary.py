import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from weread2notionpro.notion_helper import NotionHelper
from weread2notionpro.utils import get_heading

load_dotenv()

# Maximum content length (characters) sent to OpenAI to stay within token limits
MAX_CONTENT_LENGTH = 10000

# Heading text used as a marker to identify existing AI analysis sections
AI_ANALYSIS_MARKER = "AI 文献分析报告"


def _paginate_blocks(client, page_id):
    """Return all direct children blocks of a Notion page, handling pagination."""
    response = client.blocks.children.list(block_id=page_id, page_size=100)
    blocks = list(response.get("results", []))
    while response.get("has_more"):
        response = client.blocks.children.list(
            block_id=page_id,
            start_cursor=response.get("next_cursor"),
            page_size=100,
        )
        blocks.extend(response.get("results", []))
    return blocks


def extract_text_from_blocks(blocks):
    """Extract plain text from Notion content blocks (highlights and notes)."""
    text_types = {
        "callout",
        "quote",
        "paragraph",
        "bulleted_list_item",
        "numbered_list_item",
        "heading_1",
        "heading_2",
        "heading_3",
    }
    texts = []
    for block in blocks:
        block_type = block.get("type")
        if block_type in text_types:
            rich_text = block.get(block_type, {}).get("rich_text", [])
            text = "".join(rt.get("plain_text", "") for rt in rich_text)
            if text.strip():
                texts.append(text)
    return "\n".join(texts)


def find_analysis_section_index(blocks):
    """Return the index of the divider that precedes the AI analysis heading,
    or None if no analysis section is found."""
    for i, block in enumerate(blocks):
        block_type = block.get("type")
        if block_type in ("heading_1", "heading_2", "heading_3"):
            rich_text = block.get(block_type, {}).get("rich_text", [])
            text = "".join(rt.get("plain_text", "") for rt in rich_text)
            if AI_ANALYSIS_MARKER in text:
                # Include the preceding divider block if present
                start = i - 1 if i > 0 and blocks[i - 1].get("type") == "divider" else i
                return start
    return None


def analyze_with_openai(client, title, author, content):
    """Call OpenAI API to produce a structured JSON analysis of the book content."""
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    prompt = (
        f"你是一位专业的文献分析助手。请分析以下书籍《{title}》（作者：{author}）"
        "的阅读笔记和划线内容，生成一份结构化的分析报告。\n\n"
        f"以下是书籍的笔记和划线内容：\n{content}\n\n"
        "请用中文提供以下四个部分的分析，以JSON格式返回：\n"
        '1. "summary"（内容摘要）：总结文献的主要内容、核心观点和主题思想，200-400字\n'
        '2. "references"（重要文献）：列出笔记中提到的重要文献、书籍、论文、作者或理论，'
        "每条独占一行并以\"- \"开头\n"
        '3. "key_issues"（重要问题）：指出这些文献提到的重要问题和核心议题，'
        "每条独占一行并以\"- \"开头\n"
        '4. "unsolved_problems"（未解决问题）：分析可能还没有解决的潜在问题或开放性问题，'
        "每条独占一行并以\"- \"开头\n\n"
        "请确保返回有效的JSON格式，所有字段值均为字符串类型。"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=2000,
    )
    return json.loads(response.choices[0].message.content)


def _text_to_notion_blocks(text):
    """Convert a plain-text string to a list of Notion blocks.

    Lines that start with "- " become ``bulleted_list_item`` blocks; all other
    lines become ``paragraph`` blocks.  Lines longer than the Notion character
    limit are split automatically.
    """
    # Notion's rich_text content limit is 2000 characters; we use 1900 to leave
    # a safe buffer for multi-byte Unicode characters counted differently.
    MAX_BLOCK_LENGTH = 1900
    blocks = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            block_type = "bulleted_list_item"
            content = line[2:]
        else:
            block_type = "paragraph"
            content = line
        # Split oversized content
        while len(content) > MAX_BLOCK_LENGTH:
            blocks.append(
                {
                    "type": block_type,
                    block_type: {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": content[:MAX_BLOCK_LENGTH]},
                            }
                        ]
                    },
                }
            )
            content = content[MAX_BLOCK_LENGTH:]
        if content:
            blocks.append(
                {
                    "type": block_type,
                    block_type: {
                        "rich_text": [
                            {"type": "text", "text": {"content": content}}
                        ]
                    },
                }
            )
    return blocks


def build_analysis_blocks(analysis):
    """Build the list of Notion blocks that form the AI analysis report."""
    blocks = [
        {"type": "divider", "divider": {}},
        get_heading(1, f"📊 {AI_ANALYSIS_MARKER}"),
        get_heading(2, "📚 内容摘要"),
    ]
    blocks.extend(_text_to_notion_blocks(analysis.get("summary", "暂无摘要")))

    blocks.append(get_heading(2, "📖 重要文献"))
    blocks.extend(
        _text_to_notion_blocks(analysis.get("references", "未发现明显的文献引用"))
    )

    blocks.append(get_heading(2, "❓ 重要问题"))
    blocks.extend(
        _text_to_notion_blocks(analysis.get("key_issues", "未发现明显的重要问题"))
    )

    blocks.append(get_heading(2, "🔍 未解决问题"))
    blocks.extend(
        _text_to_notion_blocks(
            analysis.get("unsolved_problems", "未发现明显的未解决问题")
        )
    )
    return blocks


def update_book_analysis(notion_helper, openai_client, page_id, title, author):
    """Analyse a single book page and write (or refresh) the AI report section."""
    print(f"正在分析《{title}》...")

    blocks = _paginate_blocks(notion_helper.client, page_id)
    if not blocks:
        print(f"《{title}》没有笔记内容，跳过。")
        return

    # Locate and remove an existing analysis section so it can be refreshed
    analysis_start = find_analysis_section_index(blocks)
    if analysis_start is not None:
        for block in blocks[analysis_start:]:
            try:
                notion_helper.delete_block(block.get("id"))
            except Exception as exc:
                print(f"删除旧分析块时出错（已忽略）: {exc}")
        content_blocks = blocks[:analysis_start]
    else:
        content_blocks = blocks

    content = extract_text_from_blocks(content_blocks)
    if not content.strip():
        print(f"《{title}》没有可分析的文本内容，跳过。")
        return

    if len(content) > MAX_CONTENT_LENGTH:
        print(
            f"《{title}》内容较长（{len(content)} 字符），已截取前 {MAX_CONTENT_LENGTH} 字符进行分析，"
            "分析结果可能不完整。"
        )
        content = content[:MAX_CONTENT_LENGTH]

    try:
        analysis = analyze_with_openai(openai_client, title, author, content)
    except Exception as e:
        print(f"分析《{title}》时出错: {e}")
        return

    analysis_blocks = build_analysis_blocks(analysis)

    # Notion allows at most 100 children per append call
    batch_size = 100
    for i in range(0, len(analysis_blocks), batch_size):
        notion_helper.append_blocks(
            block_id=page_id, children=analysis_blocks[i : i + batch_size]
        )

    print(f"《{title}》分析完成。")


def _get_books_with_notes(notion_helper):
    """Return a list of dicts for books that have synced notes (Sort > 0)."""
    results = notion_helper.query_all(notion_helper.book_database_id)
    books = []
    for result in results:
        props = result.get("properties", {})

        # BookId
        book_id_prop = props.get("BookId", {})
        rich = book_id_prop.get("rich_text", [])
        book_id = rich[0].get("plain_text", "") if rich else ""

        # Title
        title_prop = props.get("书名", {})
        title_list = title_prop.get("title", [])
        title = title_list[0].get("plain_text", "") if title_list else ""

        # Author – stored as a relation; resolve each related page's title
        author = "未知作者"
        author_relations = props.get("作者", {}).get("relation", [])
        if author_relations:
            author_names = []
            for rel in author_relations:
                try:
                    page = notion_helper.client.pages.retrieve(page_id=rel["id"])
                    name_prop = page.get("properties", {}).get("标题", {})
                    name_list = name_prop.get("title", [])
                    if name_list:
                        author_names.append(name_list[0].get("plain_text", ""))
                except Exception:
                    pass
            if author_names:
                author = " ".join(author_names)

        # Sort (> 0 means notes have been synced)
        sort = props.get("Sort", {}).get("number") or 0

        if book_id and title and sort > 0:
            books.append(
                {
                    "book_id": book_id,
                    "page_id": result.get("id"),
                    "title": title,
                    "author": author,
                }
            )
    return books


def main():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise Exception(
            "未找到 OPENAI_API_KEY，请在 GitHub Secrets（或 .env 文件）中配置该变量。"
        )

    notion_helper = NotionHelper()
    openai_client = OpenAI(api_key=openai_api_key)

    books = _get_books_with_notes(notion_helper)
    print(f"找到 {len(books)} 本有笔记的书籍，开始分析…")

    for book in books:
        update_book_analysis(
            notion_helper=notion_helper,
            openai_client=openai_client,
            page_id=book["page_id"],
            title=book["title"],
            author=book["author"],
        )


if __name__ == "__main__":
    main()
