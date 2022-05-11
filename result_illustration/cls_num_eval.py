import plotly.express as px
import pandas as pd
from result import cls_eval_result as result
import plotly.graph_objects as go

l = []
for i, accs in enumerate(result["same"]):
    for j, acc in enumerate(accs):
        l.append([i + 6, j + 5, acc])
df = pd.DataFrame(l, columns=["Inserted Place", "Class Num", "Accuracy"])
print(df)

fig = go.Figure()

fig.add_trace(go.Line(df,
    x="Class Num",
    y="Accuracy",
    color="Inserted Place",
    range_y=[68, 100],
    line_shape="linear",
    markers=True,
    # width=2000,
    # height=100
))

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

fig.write_image("./output/cls_num_eval_same.png", format="png", width=700, height=500)
# fig.show()
