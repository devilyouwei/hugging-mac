import DOMPurify, { type Config } from "dompurify"
import MarkdownIt from "markdown-it"

const parser = new MarkdownIt({
  breaks: true,
  html: true,
  linkify: true,
  typographer: true,
})

const sanitizerConfig: Config = {
  FORBID_ATTR: ["style"],
  FORBID_TAGS: [
    "base",
    "button",
    "embed",
    "form",
    "iframe",
    "input",
    "link",
    "meta",
    "object",
    "option",
    "script",
    "select",
    "style",
    "textarea",
  ],
  USE_PROFILES: { html: true },
}

type Sanitize = (html: string, config: Config) => string

const sanitizeWithDOMPurify: Sanitize = (html, config) => String(DOMPurify.sanitize(html, config))

export function renderDocumentMarkdown(markdown: string, sanitize: Sanitize = sanitizeWithDOMPurify): string {
  return sanitize(parser.render(markdown), sanitizerConfig)
}
