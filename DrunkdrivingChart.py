import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import gradio as gr
import matplotlib.font_manager as fm
from shapely.geometry import Point
import io
from PIL import Image
import seaborn as sns

from config import FONTS


# Load the CSV data
def load_data():
    file_path = 'csv/NPA_TMA1_NEW.csv'
    return pd.read_csv(file_path, encoding='utf-8')


# Plotting accident distribution map using background image
def plot_accident_distribution(df):
    # Filter data for drunk driving accidents
    df = df[df['Cause_Judgment_Sub'] == '酒醉(後)駕駛']
    # Ensure latitude and longitude columns are present (case insensitive)
    df.columns = df.columns.str.lower()
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Convert latitude and longitude to numeric and drop rows with invalid values
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])

        # Create GeoDataFrame
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        geo_df = gpd.GeoDataFrame(df, geometry=geometry)

        # Set coordinate reference system (WGS 84)
        geo_df.set_crs(epsg=4326, inplace=True)

        # Load background image
        img = Image.open('image/01.臺灣地圖_文字.jpg')

        # Plot map
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, extent=[119, 122, 21.5, 25.5], aspect='auto')  # Set extent to match Taiwan coordinates

        # Plot accident distribution points
        geo_df.plot(ax=ax, markersize=10, color='red', alpha=0.7)
        ax.set_xlim([119, 122])  # Set longitude limits to ensure all data points are included
        ax.set_ylim([21.5, 25.5])  # Set latitude limits to ensure all data points are included
        plt.title("台灣車禍分布圖", fontproperties=fm.FontProperties(fname=FONTS, size=16))
        plt.xlabel("經度", fontproperties=fm.FontProperties(fname=FONTS, size=16))
        plt.ylabel("緯度", fontproperties=fm.FontProperties(fname=FONTS, size=16))

        # Save to buffer for Gradio
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return Image.open(buf)
    else:
        raise ValueError("CSV 文件中找不到 'latitude' 或 'longitude' 欄位")


# Plotting accident density map
def plot_accident_density(df):
    # Filter data for drunk driving accidents
    df = df[df['Cause_Judgment_Sub'] == '酒醉(後)駕駛']
    # Ensure latitude and longitude columns are present (case insensitive)
    df.columns = df.columns.str.lower()
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Convert latitude and longitude to numeric and drop rows with invalid values
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])

        # Create GeoDataFrame
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        geo_df = gpd.GeoDataFrame(df, geometry=geometry)

        # Set coordinate reference system (WGS 84)
        geo_df.set_crs(epsg=4326, inplace=True)

        # Load background image
        img = Image.open('image/01.臺灣地圖_文字.jpg')

        # Plot density map
        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
        ax.imshow(img, extent=[119, 122, 21.5, 25.5],
                  aspect='auto')  # Adjust extent to better align with Taiwan coordinates

        # Create hexbin heatmap to represent accident density
        x = df['longitude']
        y = df['latitude']
        hb = ax.hexbin(x, y, gridsize=50, cmap='Reds', alpha=0.7, edgecolors='white', linewidths=0.5,
                       extent=(119, 122, 21.5, 25.5))
        cbar = plt.colorbar(hb, ax=ax, orientation='vertical')
        cbar.set_label('事故密度', fontsize=12, fontweight='bold',
                       fontproperties=fm.FontProperties(fname=FONTS, size=16))
        ax.set_xlim([119, 122])  # Set longitude limits to ensure all data points are included
        ax.set_ylim([21.5, 25.5])  # Set latitude limits to ensure all data points are included
        plt.title("台灣車禍密度圖", fontsize=16, fontweight='bold',
                  fontproperties=fm.FontProperties(fname=FONTS, size=16))
        plt.xlabel("經度", fontsize=12, fontweight='bold', fontproperties=fm.FontProperties(fname=FONTS, size=16))
        plt.ylabel("緯度", fontsize=12, fontweight='bold', fontproperties=fm.FontProperties(fname=FONTS, size=16))


        # Save to buffer for Gradio
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        return Image.open(buf)
    else:
        raise ValueError("CSV 文件中找不到 'latitude' 或 'longitude' 欄位")


# Plotting Functions
def plot_pie_chart(df, save_path=None):
    drunk_driving = df[df['Cause_Judgment_Sub'] == '酒醉(後)駕駛']

    location_counts = drunk_driving['Accident_Location_Type_Sub'].value_counts()

    labels = location_counts.index
    sizes = location_counts.values
    colors = sns.color_palette("pastel")

    fig, ax = plt.subplots(figsize=(12, 10))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140,
                                      wedgeprops={'edgecolor': 'black'},
                                      textprops={'fontproperties': fm.FontProperties(fname=FONTS, size=12)})
    ax.set_title('酒駕事故地點比例', fontproperties=fm.FontProperties(fname=FONTS, size=16), pad=20)
    ax.axis('equal')  # Ensure pie chart is drawn as a circle

    # Adjust label and percentage position
    for text in texts:
        text.set_fontproperties(fm.FontProperties(fname=FONTS, size=12))
    for autotext in autotexts:
        autotext.set_fontproperties(fm.FontProperties(fname=FONTS, size=12))
        autotext.set_color('black')

    # 保存圖像到文件並返回路徑
    output_path = save_path + 'drunkdriving_pie_chart.png'
    fig.savefig(output_path, format='png', dpi=300)
    plt.close(fig)
    return output_path


def plot_bar_chart(df, save_path=None):
    drunk_driving = df[df['Cause_Judgment_Sub'] == '酒醉(後)駕駛']
    location_counts = drunk_driving['Accident_Location_Type_Sub'].value_counts()
    colors = sns.color_palette("pastel")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.bar(location_counts.index, location_counts.values, color=colors, edgecolor='black')
    ax.set_xlabel('事故地點類型', fontproperties=fm.FontProperties(fname=FONTS, size=14))
    ax.set_ylabel('酒駕事故數量', fontproperties=fm.FontProperties(fname=FONTS, size=14))
    ax.set_title('酒駕事故地點數量', fontproperties=fm.FontProperties(fname=FONTS, size=16))
    plt.xticks(rotation=45, fontproperties=fm.FontProperties(fname=FONTS, size=12))
    plt.tight_layout()

    # 保存圖像到文件並返回路徑
    output_path = save_path + 'drunkdriving_bar_chart.png'
    fig.savefig(output_path, format='png', dpi=300)
    plt.close(fig)
    return output_path


# Main Function
def main():
    # 使用 Gradio 初始化模組
    with gr.Blocks(theme=gr.themes.Monochrome(), css=".gradio-container { text-align: center; }") as demo:
        gr.Markdown(
            """
            # 酒駕數據視覺化
            生成對應的圖表來顯示酒駕數據分析。
            """
        )
        with gr.Row():
            piechart_output = gr.Image(type="filepath", label="圓餅圖", format='png')
            barchart_output = gr.Image(type="filepath", label="長條圖", format='png')
        with gr.Row():
            distribution_output = gr.Image(type="filepath", label="車禍分布圖", format='png')
            density_output = gr.Image(type="filepath", label="車禍密度圖", format='png')
        with gr.Row():
            submit_btn = gr.Button("生成車禍圖表")
            clear_btn = gr.Button("清除")

        def visualize_data():
            df = load_data()
            save_path = 'image/'
            pie_chart = plot_pie_chart(df, save_path)
            bar_chart = plot_bar_chart(df, save_path)
            distribution_map = plot_accident_distribution(df)
            density_map = plot_accident_density(df)
            return pie_chart, bar_chart, distribution_map, density_map

        def handle_input():
            return visualize_data()

        submit_btn.click(fn=handle_input, inputs=[],
                         outputs=[piechart_output, barchart_output, distribution_output, density_output])
        clear_btn.click(fn=lambda: (None, None, None, None), inputs=[],
                        outputs=[piechart_output, barchart_output, distribution_output, density_output])

    # 啟動介面
    demo.launch(share=True, server_name="0.0.0.0", server_port=7863)


if __name__ == "__main__":
    main()
