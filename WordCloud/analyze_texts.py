import pandas as pd
import jieba_fast as jieba
import glob
from collections import Counter
from snownlp import SnowNLP
import re
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 导入font_manager
import seaborn as sns
import networkx as nx
import platform
from wordcloud import WordCloud  # Added for word cloud generation

# --- 用户配置区域: 中文字体文件路径 ---
# 请将下面的路径替换为你系统上实际存在的、支持中文的字体文件路径 (.ttf, .otf, .ttc)
USER_FONT_PATH = "/Users/annfengdeye/Library/Fonts/SimHei.ttf"  # <--- *** 请务必修改为您系统上的实际字体路径 ***
# Example for other OS (if SimHei.ttf is in the same directory as the script):
# USER_FONT_PATH = "SimHei.ttf"
# Example for Windows:
# USER_FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
# Example for Linux (ensure font is installed and path is correct):
# USER_FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"


# --- 其他配置参数 ---
TOP_N_WORDS = 50
TOP_N_COOCCURRENCE = 50
TOP_N_WORDCLOUD = 100  # Max words for word cloud
SENTIMENT_POSITIVE_THRESHOLD = 0.6
SENTIMENT_NEGATIVE_THRESHOLD = 0.4
NETWORK_NODE_FIXED_SIZE = 1000  # Fixed size for network graph nodes

# --- 全局字体属性变量 ---
global_font_prop = None  # Will be set if font path is valid
networkx_font_family_name_fallback = 'sans-serif'

# --- 中文字体设置 (基于用户提供的路径) ---
print("--- 尝试配置中文字体 ---")
if USER_FONT_PATH and os.path.exists(USER_FONT_PATH):
    try:
        global_font_prop = fm.FontProperties(fname=USER_FONT_PATH)
        font_family_name_from_path = global_font_prop.get_name()
        networkx_font_family_name_fallback = font_family_name_from_path

        plt.rcParams['font.family'] = font_family_name_from_path
        # plt.rcParams['font.sans-serif'] = [font_family_name_from_path] # Redundant if font.family is set correctly
        plt.rcParams['axes.unicode_minus'] = False

        print(f"成功：已从路径 '{USER_FONT_PATH}' 加载字体 '{font_family_name_from_path}'。")
        print(f"       FontProperties 对象已创建。")
        print(f"       plt.rcParams['font.family'] 已设置为: '{font_family_name_from_path}'")
    except Exception as e:
        print(f"错误：从路径 '{USER_FONT_PATH}' 加载字体属性失败: {e}")
        print(f"       尝试直接使用字体文件名 '{os.path.basename(USER_FONT_PATH)}' 作为 Matplotlib 字体族名称。")
        try:
            font_name_direct = os.path.basename(USER_FONT_PATH).split('.')[0]
            plt.rcParams['font.family'] = font_name_direct
            plt.rcParams['axes.unicode_minus'] = False
            networkx_font_family_name_fallback = font_name_direct
            try:
                global_font_prop = fm.FontProperties(fname=USER_FONT_PATH)
                print(
                    f"       尝试使用字体名 '{font_name_direct}'。global_font_prop 仍会尝试使用路径 '{USER_FONT_PATH}'。")
            except Exception as e_fp:
                print(f"       为 global_font_prop 从路径 '{USER_FONT_PATH}' 创建 FontProperties 也失败: {e_fp}")
                global_font_prop = None
        except Exception as e2:
            print(f"       直接使用字体名也失败: {e2}")
            global_font_prop = None
            plt.rcParams['axes.unicode_minus'] = False
else:
    print(f"警告：指定的中文字体文件路径不存在或未指定: '{USER_FONT_PATH}'")
    print("       将尝试使用系统默认字体，但这通常不支持中文。")
    plt.rcParams['axes.unicode_minus'] = False
    global_font_prop = None

if global_font_prop is None:
    print("\n重要警告：未能从指定路径成功加载中文字体属性对象 (global_font_prop 为 None)。")
    print("图表中的中文（如标题、标签、刻度）可能无法正确显示。")
    print("请检查 USER_FONT_PATH 设置是否正确，并确保字体文件可用。")
print("--- 字体配置结束 ---\n")


def list_available_fonts():
    print("\n--- Matplotlib 可用字体列表 (部分) ---")
    try:
        font_names = sorted(list(set([f.name for f in fm.fontManager.ttflist])))
        for i, font_name in enumerate(font_names[:50]):
            print(f"- {font_name}")
        if len(font_names) > 50: print("  ... (及更多)")
        print("请检查您希望使用的中文字体是否在此列表中，并注意其确切名称。")
    except Exception as e:
        print(f"列出字体时出错: {e}")
    print("---------------------------------\n")


def load_stopwords(filepath="stopwords.txt"):
    stopwords = set()
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f: stopwords.add(line.strip())
    else:
        print(f"警告: 停用词文件 {filepath} 未找到。将使用基础停用词。")
        stopwords = {"的", "了", "是", "我", "你", "他", "她", "它", "们", "这", "那", "在", "有", "也", "不", "都",
                     "就"}
    return stopwords


stopwords = load_stopwords()


def load_and_combine_csvs(path="."):
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files: print("错误：未找到CSV文件。"); return pd.DataFrame()
    all_data = []
    for file_path in csv_files:
        try:
            df_temp_header = None
            try:
                df_temp_header = pd.read_csv(file_path, nrows=0, encoding='utf-8-sig')
            except UnicodeDecodeError:
                try:
                    df_temp_header = pd.read_csv(file_path, nrows=0, encoding='utf-8')
                except UnicodeDecodeError:
                    df_temp_header = pd.read_csv(file_path, nrows=0, encoding='gbk')

            if df_temp_header is not None and 'content' in df_temp_header.columns:
                df_full = None
                try:
                    df_full = pd.read_csv(file_path, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    try:
                        df_full = pd.read_csv(file_path, encoding='utf-8')
                    except UnicodeDecodeError:
                        df_full = pd.read_csv(file_path, encoding='gbk')
                if df_full is not None:
                    all_data.append(df_full)
                else:
                    print(f"警告: 文件 {file_path} 读取内容失败，已跳过。")
            else:
                print(f"警告: 文件 {file_path} 中缺少 'content' 列或无法识别编码，已跳过。")
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
    if not all_data: print("错误：所有CSV中均无有效数据或'content'列。"); return pd.DataFrame()
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"CSV合并完成。总评论数: {len(combined_df)}。")
    return combined_df


def preprocess_text(text_series, stop_words):
    processed_texts = []
    for text in text_series.astype(str).fillna(''):
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", text).lower()
        words = [w for w in jieba.lcut(text) if w.strip() and w not in stop_words and len(w) > 1]
        processed_texts.append(words)
    return processed_texts


def analyze_sentiment(text_series, font_prop):
    sentiments, scores = [], []
    if text_series.empty or text_series.isnull().all():
        return pd.DataFrame({'sentiment_score': [], 'sentiment_category': []})

    cleaned_text_series = text_series.astype(str).fillna('')

    for text in cleaned_text_series:
        if not text.strip():
            scores.append(None);
            sentiments.append("未知")
            continue
        try:
            s = SnowNLP(text);
            score = s.sentiments;
            scores.append(score)
            if score > SENTIMENT_POSITIVE_THRESHOLD:
                sentiments.append("正面")
            elif score < SENTIMENT_NEGATIVE_THRESHOLD:
                sentiments.append("负面")
            else:
                sentiments.append("中性")
        except Exception as e:
            scores.append(None);
            sentiments.append("错误")

    sentiment_df = pd.DataFrame({'sentiment_score': scores, 'sentiment_category': sentiments})

    print("\n--- 情感分析结果 ---")
    if not sentiment_df.empty and 'sentiment_category' in sentiment_df.columns:
        counts = sentiment_df['sentiment_category'].value_counts()

        custom_order = ["正面", "中性", "负面", "未知", "错误"]

        for cat in custom_order:
            if cat not in counts:
                counts[cat] = 0

        counts = counts.reindex(custom_order, fill_value=0)

        percent = (counts / counts.sum() * 100) if counts.sum() > 0 else pd.Series([0] * len(counts),
                                                                                   index=counts.index)
        summary = pd.DataFrame(
            {'数量': counts, '百分比 (%)': percent.round(2)})
        print(summary)

        plot_summary = summary[summary['数量'] > 0]
        if not plot_summary.empty:
            plt.figure(figsize=(8, 6))
            # The index of plot_summary was named 'sentiment_category' (from value_counts()).
            # reset_index() turns this named index into a column with the same name.
            # We then rename this 'sentiment_category' column to '情感类别'.
            splot_data = plot_summary.reset_index().rename(
                columns={'sentiment_category': '情感类别'})  # <-- *** FIX APPLIED HERE ***

            custom_palette = {"正面": "forestgreen", "中性": "gold", "负面": "crimson", "未知": "silver",
                              "错误": "darkgrey"}
            bar_colors = [custom_palette.get(cat, "blue") for cat in splot_data['情感类别']]

            sns.barplot(x='情感类别', y='数量', data=splot_data, palette=bar_colors, hue='情感类别', dodge=False,
                        legend=False)
            plt.title('评论情感分类统计', fontproperties=font_prop)
            plt.xlabel('情感类别', fontproperties=font_prop);
            plt.ylabel('评论数量', fontproperties=font_prop)
            for i, r in splot_data.iterrows():
                plt.text(i, r['数量'] + (splot_data['数量'].max() * 0.01 if splot_data['数量'].max() > 0 else 0.5),
                         str(int(r['数量'])), ha='center', va='bottom', fontproperties=font_prop)

            tick_labels = splot_data['情感类别']
            if font_prop:
                plt.xticks(ticks=range(len(tick_labels)), labels=tick_labels, fontproperties=font_prop)
                plt.yticks(fontproperties=font_prop)
            else:
                plt.xticks(ticks=range(len(tick_labels)), labels=tick_labels)

            plt.tight_layout();
            plt.savefig("sentiment_category_distribution.png")
            plt.close()
            print("情感分类分布条形图已保存为 sentiment_category_distribution.png")

        valid_scores = sentiment_df['sentiment_score'].dropna()
        if not valid_scores.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(valid_scores, bins=20, kde=True, color='dodgerblue', edgecolor='black')
            plt.title('情感得分分布直方图', fontproperties=font_prop)
            plt.xlabel('情感得分 (0 ≈ 负面, 1 ≈ 正面)', fontproperties=font_prop);
            plt.ylabel('评论数量 (频数)', fontproperties=font_prop)
            if font_prop: plt.xticks(fontproperties=font_prop); plt.yticks(fontproperties=font_prop)
            plt.grid(axis='y', linestyle='--', alpha=0.7);
            plt.tight_layout();
            plt.savefig("sentiment_score_histogram.png");
            plt.close()
            print("情感得分分布直方图已保存为 sentiment_score_histogram.png")
    else:
        print("未能生成情感分析摘要。")
    return sentiment_df


def word_frequency_analysis(processed_texts, font_prop):
    all_words = [w for sublist in processed_texts for w in sublist]
    if not all_words: print("警告：无词语可分析。"); return pd.DataFrame(), Counter()

    word_counts = Counter(all_words)
    top_words = word_counts.most_common(TOP_N_WORDS)

    df = pd.DataFrame(top_words, columns=['词语', '频率'])
    print(f"\n--- Top {TOP_N_WORDS} 词频分析 ---");
    print(df.to_string(index=False))
    if not df.empty:
        plt.figure(figsize=(12, max(8, len(df) * 0.4)))
        sns.barplot(x='频率', y='词语', data=df, palette="viridis", hue='词语', dodge=False, legend=False)
        plt.title(f'Top {TOP_N_WORDS} 高频词汇', fontproperties=font_prop)
        plt.xlabel('频率', fontproperties=font_prop);
        plt.ylabel('词语', fontproperties=font_prop)
        if font_prop: plt.xticks(fontproperties=font_prop); plt.yticks(fontproperties=font_prop)
        for index, value in enumerate(df['频率']):
            plt.text(value + (df['频率'].max() * 0.01 if df['频率'].max() > 0 else 0.1),
                     index,
                     str(value),
                     va='center',
                     fontproperties=font_prop)

        plt.tight_layout();
        plt.savefig("word_frequency.png");
        plt.close()
        print(f"Top {TOP_N_WORDS} 词频图已保存。")
    return df, word_counts


def semantic_network_analysis(processed_texts, word_frequency_df, font_prop, nx_font_family_fallback_name):
    from itertools import combinations
    counter = Counter(pair for wl in processed_texts if len(wl) > 1 for pair in combinations(sorted(list(set(wl))), 2))
    if not counter: print("警告：无共现词对。"); return pd.DataFrame()

    pairs = counter.most_common(TOP_N_COOCCURRENCE)
    df = pd.DataFrame([{'词语1': p[0][0], '词语2': p[0][1], '共现次数': p[1]} for p in pairs])
    print(f"\n--- Top {TOP_N_COOCCURRENCE} 共现词语 ---");
    print(df.to_string(index=False))

    if not df.empty:
        G = nx.from_pandas_edgelist(df, '词语1', '词语2', edge_attr='共现次数')
        if G.number_of_nodes() > 0:
            fig_size = max(18, int(G.number_of_nodes() * 0.5))
            plt.figure(figsize=(fig_size, fig_size))

            try:
                pos = nx.kamada_kawai_layout(G, weight='共现次数')
            except nx.NetworkXError:
                print("Kamada-Kawai布局失败，尝试使用spring_layout。")
                pos = nx.spring_layout(G, weight='共现次数',
                                       k=0.9 / (G.number_of_nodes() ** 0.5) if G.number_of_nodes() > 0 else 0.5,
                                       iterations=80, seed=42)

            node_colors_list = []
            color_palette = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#6A5ACD',
                             '#FF8C00', '#00CED1', '#ADFF2F', '#DB7093', '#F0E68C',
                             '#20B2AA', '#FF1493', '#87CEEB', '#DAA520', '#9370DB']

            ranked_words_map = {word: rank for rank, word in enumerate(word_frequency_df['词语'])}

            for node in G.nodes():
                rank = ranked_words_map.get(node)
                if rank is not None:
                    color_group = rank // 10
                    color = color_palette[color_group % len(color_palette)]
                else:
                    color = '#D3D3D3'
                node_colors_list.append(color)

            fixed_node_size = NETWORK_NODE_FIXED_SIZE

            nx.draw_networkx_nodes(G, pos,
                                   node_size=fixed_node_size,
                                   node_color=node_colors_list,
                                   alpha=0.9,
                                   edgecolors='black',
                                   linewidths=0.7)

            edge_widths_calc = [G[u][v]['共现次数'] for u, v in G.edges()]
            max_w = max(edge_widths_calc) if edge_widths_calc else 1.0
            if max_w == 0: max_w = 1.0
            edge_widths = [max(0.7, (w / max_w) * 6) + 0.3 for w in edge_widths_calc]
            if not edge_widths and G.edges(): edge_widths = [0.7] * len(G.edges())

            nx.draw_networkx_edges(G, pos, width=edge_widths,
                                   alpha=0.6,
                                   edge_color='#AAAAAA')

            num_actual_nodes = G.number_of_nodes()
            label_font_size = max(5, 12 - int(num_actual_nodes / 10))
            if fixed_node_size < 300: label_font_size = max(4, 8 - int(num_actual_nodes / 15))

            if font_prop:
                for node, (x_coord, y_coord) in pos.items():
                    plt.text(x_coord, y_coord, str(node), size=label_font_size,
                             fontproperties=font_prop,
                             ha='center', va='center', color='black',
                             bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.1)
                             )
            else:
                labels = {node: str(node) for node in G.nodes()}
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=label_font_size,
                                        font_family=nx_font_family_fallback_name, font_color='black',
                                        bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.1))

            title_font_size = max(18, int(fig_size * 0.07) + 10)
            if font_prop:
                plt.title(f'Top {TOP_N_COOCCURRENCE} 词语共现网络图', size=title_font_size, fontproperties=font_prop,
                          weight='bold')
            else:
                plt.title(f'Top {TOP_N_COOCCURRENCE} 词语共现网络图', size=title_font_size, weight='bold')

            plt.axis('off');
            plt.tight_layout(pad=0.5);
            plt.savefig("semantic_network.png", dpi=300, bbox_inches='tight');
            plt.close()
            print(f"Top {TOP_N_COOCCURRENCE} 共现网络图已保存。")
    return df


def generate_word_cloud(all_word_counts, font_path_for_wc, stopwords_for_wc, top_n=100):
    if not all_word_counts:
        print("警告：无词语可生成词云。")
        return

    top_words_for_cloud = dict(all_word_counts.most_common(top_n))

    if not top_words_for_cloud:
        print(f"警告：词频数据为空，无法生成Top {top_n}词云。")
        return

    print(f"\n--- 生成 Top {top_n} 词云图 ---")
    try:
        wc_font_path = None
        if font_path_for_wc and os.path.exists(font_path_for_wc):
            wc_font_path = font_path_for_wc
        else:
            print(f"警告: 词云字体路径 '{font_path_for_wc}' 无效或未提供。词云中的中文可能无法正确显示。")

        wc = WordCloud(font_path=wc_font_path,
                       width=1000, height=700, background_color='white',
                       max_words=top_n,
                       stopwords=stopwords_for_wc,
                       colormap='viridis',
                       collocations=False,
                       prefer_horizontal=0.95,
                       scale=2
                       )

        wc.generate_from_frequencies(top_words_for_cloud)
        wc.to_file("word_cloud.png")
        print(f"Top {top_n} 词云图已保存为 word_cloud.png")

    except Exception as e:
        print(f"生成词云时出错: {e}")
        if "بە" in str(e) or "پەت" in str(e) or "頡" in str(e):
            print(
                "错误提示包含特殊字符，这通常意味着字体不支持所有词云中的字符。请确保 USER_FONT_PATH 指向一个完整的中文字体。")
        elif "libomp" in str(e).lower():
            print("检测到 libomp 链接错误。这可能与 MacOS 上的 OpenMP 有关。尝试安装 libomp: 'brew install libomp'")


# --- 主程序 ---
if __name__ == "__main__":
    master_df = load_and_combine_csvs()
    if not master_df.empty and 'content' in master_df.columns:
        content_series = master_df['content']
        if content_series.isnull().all() or content_series.astype(str).str.strip().eq('').all():
            print("\n'content' 列为空或仅包含空白字符串，无法进行分析。")
            word_frequency_df = pd.DataFrame()
            semantic_network_df = pd.DataFrame()
            master_df['sentiment_score'] = None
            master_df['sentiment_category'] = "未知"
        else:
            print("\n开始文本预处理...")
            master_df['processed_content'] = preprocess_text(content_series, stopwords)
            print("文本预处理完成。")

            sentiment_results_df = analyze_sentiment(content_series, global_font_prop)
            master_df = pd.concat([master_df.reset_index(drop=True), sentiment_results_df.reset_index(drop=True)],
                                  axis=1)

            if 'processed_content' in master_df and not master_df['processed_content'].apply(
                    lambda x: not x if isinstance(x, list) else True).all():
                word_frequency_df, all_word_counts = word_frequency_analysis(
                    master_df['processed_content'].tolist(),
                    global_font_prop
                )

                semantic_network_df = semantic_network_analysis(
                    master_df['processed_content'].tolist(),
                    word_frequency_df,
                    global_font_prop,
                    networkx_font_family_name_fallback
                )

                generate_word_cloud(all_word_counts, USER_FONT_PATH, stopwords, top_n=TOP_N_WORDCLOUD)

            else:
                print("\n预处理后的文本内容为空，跳过词频、共现网络和词云分析。")
                word_frequency_df = pd.DataFrame()
                semantic_network_df = pd.DataFrame()
        try:
            output_filename = "text_analysis_results.xlsx"
            with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
                export_master_df = master_df.copy()
                if 'processed_content' in export_master_df.columns:
                    export_master_df['processed_content_str'] = export_master_df['processed_content'].apply(
                        lambda x: ', '.join(x) if isinstance(x, list) else '')
                    export_master_df = export_master_df.drop(columns=['processed_content'])

                export_master_df.to_excel(writer, sheet_name='所有数据及情感分析', index=False)

                if 'word_frequency_df' in locals() and not word_frequency_df.empty:
                    word_frequency_df.to_excel(writer, sheet_name='词频分析', index=False)
                if 'semantic_network_df' in locals() and not semantic_network_df.empty:
                    semantic_network_df.to_excel(writer, sheet_name='共现词语分析', index=False)
            print(f"\n所有分析结果表格已保存到: {output_filename}")
        except Exception as e:
            print(f"\n保存Excel时出错: {e}")
    else:
        print("\n未能加载数据或数据中无 'content' 列，程序终止。")

    print("\n分析完成。")