from pyecharts import options as opts
from pyecharts.charts import Line, Timeline

# 模拟从0时刻到T1时刻的数据
data_part1 = [(0, 20), (1, 35), (2, 48), (3, 55), (4, 60)]
# 模拟从0时刻到T2时刻的数据
data_part2 = data_part1 + [(5, 10), (6, 25), (7, 38), (8, 45), (9, 50)]

# 创建时间轴
timeline = Timeline()

# 添加默认展示的折线图（从0时刻到T1时刻的数据）
line_part1 = (
    Line()
    .add_xaxis([item[0] for item in data_part1])
    .add_yaxis("数据系列1", [item[1] for item in data_part1], color="blue")
    .set_global_opts(title_opts=opts.TitleOpts(title="折线图 - 数据系列1"))
)
timeline.add(line_part1, "默认展示")

# 添加通过按钮控制展示的折线图（从0时刻到T2时刻的数据）
line_part2 = (
    Line()
    .add_xaxis([item[0] for item in data_part2])
    .add_yaxis("数据系列2", [item[1] for item in data_part2], color="blue")
    .add_yaxis("数据系列1", [item[1]-0.1 for item in data_part1], color="red", is_symbol_show=False, yaxis_index=0)
    .add_yaxis("buy", [data_part1[-1][1]] * len(data_part2), color="green", 
        yaxis_index=0,
        is_smooth=False,
        is_symbol_show=False,
        linestyle_opts = opts.LineStyleOpts(type_="dashed")
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="折线图 - 数据系列2")
    )
)
timeline.add(line_part2, "按钮控制展示")

# 渲染到 HTML 文件中
timeline.render("timeline_line_chart.html")
