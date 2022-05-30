import plotly.express as px
import pandas as pd
from result import system_eval_result as result
import plotly.graph_objects as go

for scenario in ["scenario %d" % x for x in range(1, 5)]:
    for task, models in enumerate([
        ["FSHAR", "OneFi", "Ours"],
        # ["Few Radar Data", "Synthesized Data", "Fine Tune", "Radar Data"],
    ]):
        # plotly express
        # l = []
        # for model in models:
        #     accs = result[model][scenario]
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
        # fig.write_image("./output/eval_task%d_env%s.png" % (task + 1, scenario[-1]), format="png")
        # fig.show()

        # plotly go
        tick_font = 18
        title_font = 20

        fig = go.Figure(
            data=[go.Bar(
                name=model,
                x=["Env %d" % e for e in range(6 - len(result[model][scenario]), 6)],
                y=result[model][scenario]
            )
                for model in models]
        )

        fig.update_layout(
            width=700,
            height=350,
            barmode="group",
            xaxis=dict(
                # title="Environment",
                tickfont=dict(size=tick_font),
                title_font=dict(size=title_font),
            ),
            yaxis=dict(
                title="Accuracy",
                range=[0, 100],
                tickfont=dict(size=tick_font),
                title_font=dict(size=title_font),
            ),
            margin=dict(r=0, l=20, t=5, b=5),
            legend=dict(
                title_text="Model Name",
                font=dict(size=12),
                title_font=dict(size=14),
            ),
            plot_bgcolor="rgb(245, 250, 255)",
            paper_bgcolor="rgb(235, 245, 255)"
        )

        fig.write_image("./output/temp_task%d_scenario%s.png" % (task + 1, scenario[-1]), format="png")
