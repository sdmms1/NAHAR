import plotly.express as px
import pandas as pd
from result import cls_eval_result as result
import plotly.graph_objects as go

# plotly express

# l = []
# for i, accs in enumerate(result["same"]):
#     for j, acc in enumerate(accs):
#         l.append([i + 6, j + 5, acc])
# df = pd.DataFrame(l, columns=["Inserted Place", "Class Num", "Accuracy"])

# fig = px.line(
#     df,
#     x="Class Num",
#     y="Accuracy",
#     color="Inserted Place",
#     range_y=[68, 100],
#     line_shape="linear",
#     markers=True,
#     # width=2000,
#     # height=100
# )
# print(df)
# fig.write_image("./output/cls_num_eval_same.png", format="png", width=700, height=500)
# fig.show()

# plotly graph objects

tick_font = 18
title_font = 20

for tp in ["same", "diff"]:

    fig = go.Figure()
    for i, accs in enumerate(result[tp]):
        fig.add_trace(
            go.Scatter(
                x=list(range(5, 12)),
                y=accs,
                mode="lines+markers",
                name="%d" % (6 + i),
                showlegend=True,
            )
        )


    fig.update_layout(
        width=700,
        height=500,
        xaxis=dict(
            title="Number of classes",
            tickfont=dict(size=tick_font),
            title_font=dict(size=title_font),
        ),
        yaxis=dict(
            title="Accuracy",
            range=[68, 100] if tp == "diff" else [90, 100],
            tickfont=dict(size=tick_font),
            title_font=dict(size=title_font),
        ),
        margin=dict(r=5, l=5, t=5, b=5),
        legend=dict(
            title_text="Inserted Place",
            font=dict(size=18),
            title_font=dict(size=18),
        ),
        plot_bgcolor="rgb(245, 250, 255)"
    )

    fig.write_image("./output/cls_num_eval_%s.png" % tp, format="png")
