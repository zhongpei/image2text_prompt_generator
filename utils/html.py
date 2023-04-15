import html


def plaintext_to_html(text):
    text = (
            "<p>" + "<br>\n".join([f"{html.escape(x)}" for x in text.split("\n")]) + "</p>"
    )
    return text
