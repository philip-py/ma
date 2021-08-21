from ana.src.d00_utils.helper import (filter_spans_overlap,)
from ana.src.d00_helper import filter_viz
from ana.models import Doc

def render(text, row, viz, span=None, filter_by=['score'], pres=False, online=False):
    """visualize documents with displacy"""

    def filter_by_condition(viz, condition):
        viz = [i for i in viz if i[condition]]
        return viz

    viz = filter_viz(viz, on='start')
    viz = filter_spans_overlap(viz)
    viz_span = []

    if span:
        span = span
    else:
        span = (0, len(text) + 1)

    if pres:
        viz_span_ = []
        for hit in viz:
            paragraph = {}
            hit['start'] -= span[0]
            hit['end'] -= span[0]
            paragraph['start'] = hit['span_start']
            paragraph['end'] = hit['span_end']
            # hit['label'] = f"{hit['coding']} | {hit['score']:.2f}"
            if paragraph['start'] not in [i['start'] for i in viz_span_]:
                viz_span_.append(paragraph)

        for n, v in enumerate(viz_span_):
            viz_span.append({'start': v['start'], 'end': v['end'], 'label': f'P|{n+1}'})

        viz_span = sorted(viz_span, key=lambda x: x['start'])

    ##################################################
    else:

        if filter_by:
            for condition in filter_by:
                viz = filter_by_condition(viz, condition)

        if span[0] > 0:
            viz = [i for i in viz if i['span_start'] == span[0]]

        for hit in viz:

            hit['start'] -= span[0]
            hit['end'] -= span[0]

            hit['label'] = f"{hit['coding']} | {hit['score']:.2f}"
            viz_span.append(hit)

        viz_starts = set([i['span_start'] for i in viz])

        for n, start in enumerate(sorted(viz_starts)):
            if start > 0 and span[0] == 0:
                viz_span.append({'start': start-1, 'end': start, 'label': f'P{n+1} | {start}'})

        viz_span = sorted(viz_span, key=lambda x: x['start'])
    ###############################################

    if online:
        ex = [
            {
                "text": text[span[0]: span[1]],
                "ents": viz_span,
                "title": 'user-input analysis'
                # "title": f"{row['doc']} | {row.name_res} ({row['party']}) | {row['date'].strftime('%d/%m/%Y')}",
                # 'title': 'text'
            }
        ]

    else:
        ex = [
            {
                "text": text[span[0]: span[1]],
                "ents": viz_span,
                "title": f"{row['doc'][0]} | {row.name_res[0]} ({row['party'][0]}) | {row['date'][0].strftime('%d/%m/%Y')}",
                # "title": f"{row['doc']} | {row.name_res} ({row['party']}) | {row['date'].strftime('%d/%m/%Y')}",
                # 'title': 'text'
            }
        ]
    all_ents = {i["label"] for i in viz_span}

    options = {"ents": all_ents, "colors": dict()}
    for ent in all_ents:
        if ent.startswith("E"):
            options["colors"][ent] = "coral"
        if ent.startswith("V"):
            options["colors"][ent] = "lightgrey"
        if ent.startswith("P"):
            options["colors"][ent] = "yellow"
    print(viz_span)
    return (ex, options)


def timeseries(res):
    viz = res.df.groupby(["party"]).resample("Q").mean().reset_index()
    viz.drop(
        viz[(viz["party"] == "Parteilos") | (viz["party"] == "Die blaue Partei")].index,
        inplace=True,
    )
    viz2 = res.df.resample("Q").mean().reset_index()
    import plotly.express as px
    import plotly.graph_objects as go

    fig = px.line(x=viz.date, y=viz.score, color=viz.party, title="Populism-Score")
    fig.update_layout(width=1000, title_font_size=20)

    colors = ["darkblue", "black", "blue", "green", "magenta", "gold", "red"]
    for j, party in enumerate([i.name for i in fig.data]):
        fig.data[j].line.color = colors[j]
    fig.add_trace(
        go.Scatter(
            x=viz2.date,
            y=viz2.score,
            mode="lines",
            name="Durchschnit / Quartal",
            marker_symbol="pentagon",
            line_width=5,
            line_color="darkgrey",
            line_dash="dash",
        )
    )
    fig.layout.template = "plotly_white"
    fig.layout.legend.title.text = "Partei"
    fig.layout.xaxis.title.text = "Jahr"
    fig.layout.yaxis.title.text = "Score"
    fig.update_traces(hovertemplate="Score: %{y} <br>Jahr: %{x}")
    for i in range(2, 7):
        fig.data[i].visible = "legendonly"
    return fig

def table(df):
    df.sort_values(by='score', ascending=False, inplace=True)
    df['datum'] = df['date'].apply(lambda x: x.strftime('%d-%m-%Y'))
    # df['A'].apply(lambda x: x.strftime('%d%m%Y'))
    cols = ['name_res', 'party', 'datum', 'score']
    table = go.Figure([go.Table(
        header=dict(
            values=cols,
            font=dict(size=12),
            align="left"
        ),
        cells=dict(
            values=[df[k].tolist() for k in cols],
            align = "left")
        )
    ])

    table.update_layout(
        autosize=True,
    )
    return table


