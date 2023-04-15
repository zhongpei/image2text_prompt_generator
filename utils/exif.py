import piexif
import piexif.helper
from .html import plaintext_to_html


def get_image_info(rawimage):
    items = rawimage.info
    geninfo = ""

    if "exif" in rawimage.info:
        exif = piexif.load(rawimage.info["exif"])
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode("utf8", errors="ignore")

        items["exif comment"] = exif_comment
        geninfo = exif_comment

        for field in [
            "jfif",
            "jfif_version",
            "jfif_unit",
            "jfif_density",
            "dpi",
            "exif",
            "loop",
            "background",
            "timestamp",
            "duration",
        ]:
            items.pop(field, None)

    geninfo = items.get("parameters", geninfo)

    info = f"""
    <p><h4>PNG Info</h4></p>
    """
    for key, text in items.items():
        info += (
                f"""
    <div>
    <p><b>{plaintext_to_html(str(key))}</b></p>
    <p>{plaintext_to_html(str(text))}</p>
    </div>
    """.strip()
                + "\n"
        )

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"
    return info
