import docutils.core

source = "usage.rst"
output = "usage.html"

docutils.core.publish_file(
    source_path = source,
    destination_path = output,
    writer_name="html"
)