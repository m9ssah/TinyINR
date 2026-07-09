import katex from "katex";

interface KatexInlineProps {
    expression: string;
}

export function KatexInline({ expression }: KatexInlineProps) {
    const html = katex.renderToString(expression, {
        displayMode: false,
        throwOnError: false,
        output: "html",
    });

    return <span dangerouslySetInnerHTML={{ __html: html }} />;
}