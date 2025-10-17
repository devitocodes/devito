import json
import sys
from string import Template


def main():
    error_code = 0

    # Standard Github message template for CI annotations
    message_template = Template(
        '::error file=$path,line=$line_num,col=$byte_offset,endcol=$end_col,'
        'title=$type::`$typo` should be $suggestions'
    )

    for line in sys.stdin:
        error_code = 1
        # Grab the JSON data coming from typos from stdin
        data = json.loads(line.rstrip())

        if data['type'] == 'binary_file':
            continue

        try:
            # Calculate the end column and format the correction
            suggestions = ', '.join(data['corrections'])
            end_col = data['byte_offset'] + len(data['typo'])

            # Print the templated message to stdout
            print(message_template.safe_substitute(
                data, end_col=end_col, suggestions=suggestions
            ))
        except KeyError:
            print('KeyError')
            print(f'{data}')
        except Exception as e:
            print('Caught unhandled exception')
            print(f'{data}')
            print(f'{e}')

    return error_code


if __name__ == '__main__':
    exit(main())
