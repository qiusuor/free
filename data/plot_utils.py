from pyecharts.charts import Kline, Bar, Line, Grid
import pyecharts.options as opts
from pyecharts.commons.utils import JsCode
import joblib


def plot_kline_volume(data, title, html):
    kline = (
        Kline(init_opts=opts.InitOpts(width="1400px", height="650px"))
        .add_xaxis(xaxis_data=list(data.index))
        .add_yaxis(
            series_name="klines",
            y_axis=data[["open", "close", "low", "high"]].values.tolist(),
            itemstyle_opts=opts.ItemStyleOpts(
                color="#ec0000", color0="#00da3c"),
        )
        .set_global_opts(legend_opts=opts.LegendOpts(is_show=True, pos_bottom=10, pos_left="center"),
                         datazoom_opts=[
            opts.DataZoomOpts(
                is_show=False,
                type_="inside",
                xaxis_index=[0, 1],
                range_start=98,
                range_end=100,
            ),
            opts.DataZoomOpts(
                is_show=True,
                xaxis_index=[0, 1],
                type_="slider",
                pos_top="85%",
                range_start=98,
                range_end=100,
            ),
        ],
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
        ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                background_color="rgba(245, 245, 245, 0.8)",
                border_width=1,
                border_color="#ccc",
                textstyle_opts=opts.TextStyleOpts(color="#000"),
        ),
            visualmap_opts=opts.VisualMapOpts(
                is_show=False,
                dimension=2,
                series_index=5,
                is_piecewise=True,
                pieces=[
                    {"value": 1, "color": "#00da3c"},
                    {"value": -1, "color": "#ec0000"},
                ],
        ),
            axispointer_opts=opts.AxisPointerOpts(
                is_show=True,
                link=[{"xAxisIndex": "all"}],
                label=opts.LabelOpts(background_color="#777"),
        ),
            brush_opts=opts.BrushOpts(
                x_axis_index="all",
                brush_link="all",
                out_of_brush={"colorAlpha": 0.1},
                brush_type="lineX",
        ),
        )
    )

    bar = (
        Bar()
        .add_xaxis(xaxis_data=list(data.index))
        .add_yaxis(
            series_name="turn",
            y_axis=data["turn"].tolist(),
            xaxis_index=1,
            yaxis_index=1,
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(
                color=JsCode(
                    """
                function(params) {
                    var colorList;
                    if (barData[params.dataIndex][1] > barData[params.dataIndex][0]) {
                        colorList = '#ef232a';
                    } else {
                        colorList = '#14b143';
                    }
                    return colorList;
                }
                """
                )
            ),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                grid_index=1,
                axislabel_opts=opts.LabelOpts(is_show=False),
            ),
            legend_opts=opts.LegendOpts(is_show=False),
        )
    )

    grid_chart = Grid(
        init_opts=opts.InitOpts(
            width="1400px",
            height="650px",
            animation_opts=opts.AnimationOpts(animation=False),
        )
    )

    grid_chart.add_js_funcs("var barData={}".format(
        data[["open", "close"]].values.tolist()))
    grid_chart.add(
        kline,
        grid_opts=opts.GridOpts(pos_left="5%", pos_right="5%", height="40%"),
    )
    grid_chart.add(
        bar,
        grid_opts=opts.GridOpts(
            pos_left="5%", pos_right="5%", pos_top="60%", height="20%"
        ),
    )
    grid_chart.render(html)


if __name__ == "__main__":
    data = joblib.load("/Users/yuqiang/workspace/freedom/data/data/daily/sh.600172_d_2.pkl")
    plot_kline_volume(data, title='600172', html="kline_volume_signal.html")
