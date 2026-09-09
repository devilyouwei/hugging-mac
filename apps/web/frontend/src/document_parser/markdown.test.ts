import { describe, expect, it, vi } from "vitest"
import { renderDocumentMarkdown } from "./markdown"

describe("renderDocumentMarkdown", () => {
  it("passes embedded document HTML through the Markdown renderer and sanitizer", () => {
    const sanitize = vi.fn((html: string, _config: unknown) => html)

    const rendered = renderDocumentMarkdown(
      "<table><tr><td>Cell</td></tr></table>\n\nWater is H<sub>2</sub>O.",
      sanitize,
    )

    expect(rendered).toContain("<table>")
    expect(rendered).toContain("H<sub>2</sub>O")
    expect(sanitize).toHaveBeenCalledOnce()
    expect(sanitize.mock.calls[0]?.[1]).toMatchObject({
      FORBID_ATTR: ["style"],
      USE_PROFILES: { html: true },
    })
  })

  it("sends unsafe raw HTML through the sanitizer", () => {
    const sanitize = vi.fn((html: string, _config: unknown) =>
      html.replace(/<script[\s\S]*?<\/script>/gi, "").replace(/\sonerror=(?:"[^"]*"|'[^']*')/gi, ""),
    )

    const rendered = renderDocumentMarkdown(
      '<p>Safe</p><script>alert(1)</script><img src="x" onerror="alert(2)">',
      sanitize,
    )

    expect(rendered).toContain("<p>Safe</p>")
    expect(rendered).not.toContain("<script")
    expect(rendered).not.toContain("onerror")
  })
})
