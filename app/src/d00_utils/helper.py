import os
from dataclasses import dataclass
from typing import List, Text
from app.config import config

def get_main_dir():
    BASE_DIR = os.path.join(config[os.getenv('FLASK_CONFIG')].DIR_MAIN)
    return BASE_DIR

def get_data_dir():
    DATA_DIR = os.path.join(config[os.getenv('FLASK_CONFIG')].DIR_DATA)
    return DATA_DIR

def get_res_dir():
    RES_DIR = os.path.join(config[os.getenv('FLASK_CONFIG')].DIR_RES)
    return RES_DIR


def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


def flatten(arr):
    """Flatten array."""
    for i in arr:
        if isinstance(i, list):
            yield from flatten(i)
        elif isinstance(i, set):
            yield from flatten(i)
        else:
            yield i


def filter_spans_overlap(spans):
    """Filter a sequence of spans and remove duplicates AND DIVIDE!!! overlaps. Useful for
    creating named entities (where one token can only be part of one entity) or
    when merging spans with `Retokenizer.merge`. When spans overlap, the (first)
    longest span is preferred over shorter spans.

    spans (iterable): The spans to filter.
    RETURNS (list): The filtered spans.
    """
    # get_sort_key = lambda span: (span['span_end'] - span['span_start'], -span['span_start'])
    # sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    sorted_spans = sorted(spans, key=lambda span: span['span_start'])
    # print(sorted_spans)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # current_start = span['span_start']
        # current_end = span['span_end']
        # current_stop =
        # Check for end - 1 here because boundaries are inclusive
        if span['span_start'] in seen_tokens:
            for s in result:
                if s['span_start'] == result[-1]['span_start']:
                    if s['span_end'] < span['span_end']:
                        s['span_end'] = span['span_end']
                    else:
                        span['span_end'] = s['span_end']

            span['span_start'] = result[-1]['span_start']
            result.append(span)

        elif span['span_start'] not in seen_tokens and span['span_end']- 1 not in seen_tokens:
            result.append(span)


        seen_tokens.update(range(span['span_start'], span['span_end']))
    result = sorted(result, key=lambda span: span['span_start'])
    return result


def filter_spans_overlap_no_merge(spans):
    sorted_spans = sorted(spans, key=lambda span: span['span_start'])
    result = []
    seen_tokens = set()
    seen_starts = dict()
    for span in sorted_spans:
        span_len = span['span_end'] - span['span_start']
        # Check for end - 1 here because boundaries are inclusive
        if span['span_start'] not in seen_starts and span['span_end'] not in seen_tokens:
            seen_starts[span['span_start']] = span_len
            result.append(span)

        elif span['span_start'] in seen_starts:
            if span['span_end']-1 not in seen_tokens:
                if span_len > seen_starts[span['span_start']]:
                    seen_starts[span['span_start']] = span_len
                    for r in result:
                        if r['span_start'] == span['span_start']:
                            r['span_end'] = span['span_end']


                # if s['span_start'] == result[-1]['span_start']:
                #     if s['span_end'] < span['span_end']:
                #         s['span_end'] = span['span_end']
                #     else:
                #         span['span_end'] = s['span_end']

            # span['span_start'] = result[-1]['span_start']
            # result.append(span)

        elif span['span_start'] not in seen_tokens and span['span_end']- 1 not in seen_tokens:
            result.append(span)

        seen_tokens.update(range(span['span_start'], span['span_end']))
    result = sorted(result, key=lambda span: span['span_start'])
    return result


def filter_viz(viz, on='start'):
    res = []
    ids = set()
    for hit in viz:
        if hit[on] not in ids:
            res.append(hit)
            ids.add(hit[on])

    return res


# Simple tests to check basic setup

def add_1(x):
    return x+1

def sub_1(x):
    return x-1

def divide_by_zero(x):
    res = x / 0
    return res

@dataclass
class Person:
    name: Text

    @property
    def name_as_first_and_last(self) -> list:
        return self.name.split()


# # backup of data cleaning
# from app.src.d00_utils.helper import get_main_dir, get_data_dir
# import pandas as pd

# DIR_META = os.path.join(get_data_dir(), "mdbs_metadata.pkl")
# DIR_CORPUS = os.path.join(get_data_dir(), "corpus")

# df = pd.read_json(f'{DIR_CORPUS}/presse_meta.json', orient='index')

# #for twitter
# df.rename(columns={'timestamp':'datum'}, inplace=True)

# df["date"] = pd.to_datetime(df["datum"])

# df.to_json('presse_meta.json', orient='index')

# # count nans
# df.groupby('party').agg({"presse_datum_clean": lambda x: x.isnull().sum()}).rename(columns={"presse_datum_clean":"count"}).sort_values(by='count')
