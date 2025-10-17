import json
import sys
from string import Template


def main():
    # Standard Github message template for CI annotations
    message_template = Template(
        '::error file=$path,line=$line_num,col=$byte_offset,endcol=$end_col,'
        'title=$type::`$typo` should be $suggestions'
    )

    for line in sys.stdin:
        # Grab the JSON data coming from typos from stdin
        data = json.loads(line.rstrip())

        # Calculate the end column and format the correction
        end_col = data['byte_offset'] + len(data['typo'])
        suggestions = ', '.join(data['corrections'])

        # Print the templated message to stdout
        print(message_template.safe_substitute(
            data, end_col=end_col, suggestions=suggestions
        ))


if __name__ == '__main__':
    exit(main())
