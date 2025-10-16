import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # for 3D bar if needed
import numpy as np
import matplotlib

# Fix emoji rendering
matplotlib.rcParams['font.family'] = 'Segoe UI Emoji'

st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Done by Rohan B")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')

    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):

        # ----------------- Top Statistics -----------------
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"<div style='text-align:center; font-size:20px;'><b>Total Messages</b><br><span style='font-size:28px;'><b>{num_messages}</b></span></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div style='text-align:center; font-size:20px;'><b>Total Words</b><br><span style='font-size:28px;'><b>{words}</b></span></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div style='text-align:center; font-size:20px;'><b>Media Shared</b><br><span style='font-size:28px;'><b>{num_media_messages}</b></span></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div style='text-align:center; font-size:20px;'><b>Links Shared</b><br><span style='font-size:28px;'><b>{num_links}</b></span></div>", unsafe_allow_html=True)

        # ----------------- Monthly Timeline -----------------
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # ----------------- Daily Timeline -----------------
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # ----------------- Activity Map -----------------
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # ----------------- Most Busy Users (Group-level) -----------------
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            new_df.rename(columns={'percent': 'members'}, inplace=True)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # ----------------- WordCloud -----------------
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # ----------------- Most Common Words -----------------
        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(most_common_df[0], most_common_df[1])
        plt.xticks(rotation='vertical')
        st.title('Most common words')
        st.pyplot(fig)

        # ----------------- Emoji Analysis -----------------
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
            st.pyplot(fig)

        # ----------------- Link Analysis (Hollow Pie) -----------------
        st.title("Link Analysis")
        link_df = helper.link_analysis(selected_user, df)
        colors = plt.cm.tab20.colors[:len(link_df)]
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            link_df['Count'],
            labels=None,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops={'width':0.4}
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
        ax.legend(
            wedges,
            link_df['Domain'],
            title="Domains",
            loc='lower center',
            bbox_to_anchor=(0.5, -0.15),
            fontsize=8,
            title_fontsize=10,
            ncol=3
        )
        ax.set(aspect="equal")
        st.pyplot(fig)

        # ----------------- Sentiment Analysis (Bar Graph, % Perspective) -----------------
        st.title("Sentiment Analysis")
        model, vectorizer = preprocessor.train_sentiment_model(df)
        sentiment_df = helper.sentiment_analysis(selected_user, df, model, vectorizer)

        # Convert to percentage
        sentiment_df['Percentage'] = (sentiment_df['Count'] / sentiment_df['Count'].sum()) * 100

        fig, ax = plt.subplots()
        colors = ['green', 'orange', 'red'][:len(sentiment_df)]
        ax.bar(sentiment_df['Sentiment'], sentiment_df['Percentage'], color=colors)
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("Sentiment")
        ax.set_title("Sentiment Analysis")
        for i, val in enumerate(sentiment_df['Percentage']):
            ax.text(i, val + 1, f"{val:.1f}%", ha='center', fontsize=10)
        st.pyplot(fig)
