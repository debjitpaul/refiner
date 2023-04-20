# coding=utf-8

from __future__ import print_function
import re
import ast
import astor


QUOTED_STRING_RE = re.compile(r"(?P<quote>[`'\"])(?P<string>.*?)(?P=quote)")


def canonicalize_intent(intent):
    str_matches = QUOTED_STRING_RE.findall(intent)

    slot_map = dict()

    return intent, slot_map


def replace_strings_in_ast(py_ast, string2slot):
    for node in ast.walk(py_ast):
        for k, v in list(vars(node).items()):
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            # Python 3
            # if isinstance(v, str) or isinstance(v, unicode):
            if isinstance(v, str):
                if v in string2slot:
                    val = string2slot[v]
                    # Python 3
                    # if isinstance(val, unicode):
                    #     try: val = val.encode('ascii')
                    #     except: pass
                    setattr(node, k, val)
                else:
                    # Python 3
                    # if isinstance(v, str):
                    #     str_key = unicode(v)
                    # else:
                    #     str_key = v.encode('utf-8')
                    str_key = v

                    if str_key in string2slot:
                        val = string2slot[str_key]
                        if isinstance(val, str):
                            try: val = val.encode('ascii')
                            except: pass
                        setattr(node, k, val)


def canonicalize_code(code, slot_map):
    string2slot = {x[1]['value']: x[0] for x in list(slot_map.items())}

    py_ast = ast.parse(code)
    replace_strings_in_ast(py_ast, string2slot)
    canonical_code = astor.to_source(py_ast)

    return canonical_code


def decanonicalize_code(code, slot_map):
    try:
      slot2string = {x[0]: x[1]['value'] for x in list(slot_map.items())}
      py_ast = ast.parse(code)
      replace_strings_in_ast(py_ast, slot2string)
      raw_code = astor.to_source(py_ast)
      # for slot_name, slot_info in slot_map.items():
      #     raw_code = raw_code.replace(slot_name, slot_info['value'])

      return raw_code.strip()
    except:
      return code
