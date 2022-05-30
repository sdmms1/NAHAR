import plotly.express as px
import pandas as pd
from result import synthesized_as_support_result as result
import plotly.graph_objects as go

# plotly express

# l = []
# for model in result.keys():
#     if model == "Radar":
#         continue
#     accs = result[model]
#     for i, acc in enumerate(accs):
#         l.append([model, "Env %d" % (6 - len(accs) + i), acc])
# df = pd.DataFrame(l, columns=["Model", "Environment", "Accuracy"])
# print(df)
#
# fig = px.bar(
#     df,
#     x="Environment",
#     y="Accuracy",
#     color="Model",
#     barmode="group",
#     range_y=[0, 100]
# )
#
# fig.write_image("./output/support_from_synthesized.png", format="png", width=700, height=500)
# fig.show()

# plotly graph object

tick_font = 15
title_font = 20

fig = go.Figure(
    data=[go.Bar(name=model, x=["Environment %d" % e for e in range(1, 6)], y=result[model])
          for model in result.keys() if model != "Radar"]
)


fig.update_layout(
    width=1000,
    height=600,
    barmode="group",
    xaxis=dict(
        title="Environment",
        tickfont=dict(size=tick_font),
        title_font=dict(size=title_font),
    ),
    yaxis=dict(
        title="Accuracy",
        range=[0, 100],
        tickfont=dict(size=tick_font),
        title_font=dict(size=title_font),
    ),
    margin=dict(r=5, l=5, t=5, b=5),
    legend=dict(
        title_text="Model Name",
        font=dict(size=12),
        title_font=dict(size=12),
    ),
    plot_bgcolor="rgb(245, 250, 255)"
)

fig.write_image("./output/support_from_synthesized.png", format="png")

