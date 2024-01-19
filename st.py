import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.ticker as ticker
import Data_Preprocessing , Load_Functions , Load_Model

st.sidebar.title("Select One")
app_selection = st.sidebar.selectbox("Select App", ["Prediction Using Whatsapp Dataset", "User Input Prediction"])

if app_selection == "Prediction Using Whatsapp Dataset":
    uploaded_file =st.sidebar.file_uploader("Upload your file")

    st.markdown("<h1 style='text-align: center; color: LightSlateGray;'>English WhatsApp Chat Sentiment Analyzer</h1>", unsafe_allow_html=True)

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        d = bytes_data.decode("utf-8")
        data = Data_Preprocessing.preprocess(d)
        data = data.apply(Load_Functions.ct, axis=1)
        data = data[data['Cleaned_Text'] != ''].reset_index(drop=True)
        data = data.drop_duplicates('Cleaned_Text').reset_index(drop=True)
        X = data.dropna()['Cleaned_Text']
        pred = Load_Functions.predict_sentiment_for_column(X, Load_Model.xgb, Load_Model.vectorizer)
        data['value'] = pred
        data['Sentiment'] = data.apply(lambda row: Load_Functions.sentiment(row), axis=1)
        st.dataframe(data)

        # user name list
        user_list = data['User'].iloc[1:].unique().tolist()
        # sorting
        user_list.sort()
        # insert "Overall" at index 0
        user_list.insert(0, "Overall")
        # Selectbox
        selected_user = st.sidebar.selectbox("Choose a user", user_list)
        if st.sidebar.button("Show Analysis"):
            st.markdown(f"<div style='font-size: 1.2em;'><span style='color: black;'>Statistics for: </span><span style='color: LightSlateGray;'>"
                        f"{selected_user}</span></div>", unsafe_allow_html=True)
            u = len(data['User'].iloc[1:].unique().tolist())
            st.markdown(f"<div style='font-size: 1.2em;'><span style='color: black;'>Total members: </span><span style='color: LightSlateGray;'>"
                        f"{u}</span></div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            start_dt = str(data['Only_Date'].iloc[0])[:10]
            last_dt = str(data['Only_Date'].iloc[-1])[:10]
            with col1:
                st.markdown(f"<div style='font-size: 1.2em;'><span style='color: black;'>Chat from: </span><span style='color: LightSlateGray;'>"
                            f"{start_dt}</span></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='font-size: 1.2em;'><span style='color: black;'>Chat to: </span><span style='color: LightSlateGray;'>"
                            f"{last_dt}</span></div>", unsafe_allow_html=True)

            total_messages, total_words = Load_Functions.fun_stats(selected_user, data)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='font-size: 1.2em;'><span style='color: black;'>Total messages: </span><span style='color: LightSlateGray;"
                            f"'>{total_messages}</span></div>", unsafe_allow_html=True)
                #st.title(total_messages)
            with col2:
                st.markdown(f"<div style='font-size: 1.2em;'><span style='color: black;'>Total words: </span><span style='color: LightSlateGray;'>"
                            f"{total_words}</span></div>", unsafe_allow_html=True)
                #st.title(total_words)

            st.write("")
            st.write("")

            st.header("Timeline Analysis")
            col1, col2 = st.columns(2)
            with col2:
                # monthly timeline
                st.markdown("<h3 style='text-align: center;'>Monthly Timeline</h3>",
                        unsafe_allow_html=True)
                timeline = Load_Functions.monthly_timeline(selected_user, data,1)
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['Cleaned_Text'], color='MediumAquamarine')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col1:
                # daily timeline
                st.markdown("<h3 style='text-align: center;'>Daily Timeline</h3>",
                        unsafe_allow_html=True)
                daily_timeline = Load_Functions.daily_timeline(selected_user, data,1)
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['Only_Date'], daily_timeline['Cleaned_Text'], color='LightSeaGreen')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # activity map
            st.header("Activity Map")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h3 style='text-align: center; '>Most busy day</h3>",
                        unsafe_allow_html=True)
                busy_day = Load_Functions.week_activity_map(selected_user, data, 1)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='LightSeaGreen')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.markdown("<h3 style='text-align: center;'>Most busy month</h3>",
                        unsafe_allow_html=True)
                busy_month = Load_Functions.month_activity_map(selected_user, data, 1)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='MediumAquamarine')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)


            period = []
            for hour in data[['Day_name', 'Hour']]['Hour']:
                if hour == 23:
                    period.append(str(hour) + "-" + str('00'))
                elif hour == 0:
                    period.append(str('00') + "-" + str(hour + 1))
                else:
                    period.append(str(hour) + "-" + str(hour + 1))

            if selected_user == 'Overall':
                st.header("Most Busy Users")
                x, new_df = Load_Functions.most_busy_users(data)
                fig, ax = plt.subplots()

                col1, col2 = st.columns(2)

                with col1:
                    ax.bar(x.index, x.values, color='DarkCyan')
                    plt.xticks()
                    st.pyplot(fig)

                with col2:
                    st.dataframe(new_df)

            # pie chart of user activity percentage
            st.header("Users Activity in Percentage")
            user_count = data['User'].iloc[1:].value_counts().reset_index()
            user_count.columns = ['member', 'text']
            fig, ax = plt.subplots()
            ax.pie(user_count['text'], labels=user_count['member'] , autopct='%1.1f%%', startangle=90,
               colors=plt.cm.Paired.colors)
            ax.axis('equal')
            ax.legend(user_count['member'], bbox_to_anchor=(1.05, 1), loc='upper left')
            st.pyplot(fig)

            st.header("Wordcloud for All the Sentiments")
            st.subheader('Word cloud')
            st.write('Top 50 words in all the chat represented as word cloud')
            fig,ax = plt.subplots()
            ax = Load_Functions.get_word_cloud(data)
            st.pyplot(fig)

            st.subheader('Word cloud positive')
            st.write('Top 50 potisive words in all the chat represented as word cloud')
            fig, ax = plt.subplots()
            ax = Load_Functions.get_word_cloud_positive(data)
            st.pyplot(fig)

            st.subheader('Word cloud neutral')
            st.write('Top 50 neutral words in all the chat represented as word cloud')
            fig,ax = plt.subplots()
            ax = Load_Functions.get_word_cloud_neutral(data)
            st.pyplot(fig)

            st.subheader('Word cloud negative')
            st.write('Top 50 words negative in all the chat represented as word cloud')
            fig,ax = plt.subplots()
            ax = Load_Functions.get_word_cloud_negative(data)
            st.pyplot(fig)

            # Percentage contributed
            st.header("User Percentage Contributions")
            if selected_user == 'Overall':
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Most Positive Contribution</h3>",
                            unsafe_allow_html=True)
                    x = Load_Functions.percentage(data, 2)
                    # Displaying
                    st.dataframe(x)
                with col2:
                    st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Most Neutral Contribution</h3>",
                            unsafe_allow_html=True)
                    y = Load_Functions.percentage(data, 1)
                    # Displaying
                    st.dataframe(y)
                with col3:
                    st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Most Negative Contribution</h3>",
                            unsafe_allow_html=True)
                    z = Load_Functions.percentage(data, 0)
                    # Displaying
                    st.dataframe(z)

            # Most Positive,Negative,Neutral User...
            if selected_user == 'Overall':
                # Getting names per sentiment
                x = data['User'][data['value'] == 1].value_counts().head(10)
                y = data['User'][data['value'] == 2].value_counts().head(10)
                z = data['User'][data['value'] == 0].value_counts().head(10)
                col1, col2, col3 = st.columns(3)
                with col1:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Most Positive Users</h3>",
                            unsafe_allow_html=True)
                    # Displaying
                    fig, ax = plt.subplots()
                    ax.bar(y.index, y.values, color='darkcyan')
                    plt.xticks()
                    st.pyplot(fig)
                with col2:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Most Neutral Users</h3>",
                            unsafe_allow_html=True)
                    # Displaying
                    fig, ax = plt.subplots()
                    ax.bar(x.index, x.values, color='lightseagreen')
                    plt.xticks()
                    st.pyplot(fig)
                with col3:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Most Negative Users</h3>",
                            unsafe_allow_html=True)

                    # Displaying
                    fig, ax = plt.subplots()
                    ax.bar(z.index, z.values, color='turquoise')
                    plt.xticks()
                    st.pyplot(fig)

                    # Monthly activity map
                st.header("Monthly Activity Map for Each Sentiment")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Positive</h3>",
                                unsafe_allow_html=True)

                    busy_month = Load_Functions.month_activity_map(selected_user, data, 2)

                    fig, ax = plt.subplots()
                    ax.bar(busy_month.index, busy_month.values, color='darkcyan')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Neutral</h3>",
                                unsafe_allow_html=True)
                    busy_month = Load_Functions.month_activity_map(selected_user, data, 1)
                    fig, ax = plt.subplots()
                    ax.bar(busy_month.index, busy_month.values, color='lightseagreen')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col3:
                    st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Negative</h3>",
                                unsafe_allow_html=True)
                    busy_month = Load_Functions.month_activity_map(selected_user, data, 0)
                    fig, ax = plt.subplots()
                    ax.bar(busy_month.index, busy_month.values, color='turquoise')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
            # Daily activity map
            st.header("Daily Activity Map for Each Sentiment")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Positive</h3>",
                        unsafe_allow_html=True)
                busy_day = Load_Functions.week_activity_map(selected_user, data, 2)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='darkcyan')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Neutral</h3>",
                        unsafe_allow_html=True)
                busy_day = Load_Functions.week_activity_map(selected_user, data, 1)
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='lightseagreen')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Negative</h3>",
                        unsafe_allow_html=True)

                busy_day = Load_Functions.week_activity_map(selected_user, data, 0)

                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='turquoise')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Daily timeline
            st.header("Daily Timeline for Each Sentiment")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Positive</h3>",
                        unsafe_allow_html=True)

                daily_timeline = Load_Functions.daily_timeline(selected_user, data, 2)
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['Only_Date'], daily_timeline['Cleaned_Text'], color='darkcyan')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Neutral</h3>",
                        unsafe_allow_html=True)
                daily_timeline = Load_Functions.daily_timeline(selected_user, data, 1)
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['Only_Date'], daily_timeline['Cleaned_Text'], color='lightseagreen')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Negative</h3>",
                        unsafe_allow_html=True)
                daily_timeline = Load_Functions.daily_timeline(selected_user, data, 0)
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['Only_Date'], daily_timeline['Cleaned_Text'], color='turquoise')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            # Monthly timeline
            st.header("Monthly Timeline for Each Sentiment")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Positive</h3>",
                        unsafe_allow_html=True)
                timeline = Load_Functions.monthly_timeline(selected_user, data, 2)
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['Cleaned_Text'], color='darkcyan')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Neutral</h3>",
                        unsafe_allow_html=True)
                timeline = Load_Functions.monthly_timeline(selected_user, data, 1)
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['Cleaned_Text'], color='lightseagreen')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;font-size: 18px;'>Negative</h3>",
                        unsafe_allow_html=True)

                timeline = Load_Functions.monthly_timeline(selected_user, data, 0)

                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['Cleaned_Text'], color='turquoise')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

else:
    st.markdown("<h1 style='text-align: center; color: LightSlateGray;'>English WhatsApp Chat Sentiment Prediction</h1>",
                unsafe_allow_html=True)
    user_input = st.text_input("Enter a sentence:")

    # Display the entered sentence
    # if user_input:
    #     st.write(f"You entered: {user_input}")


    result1 = Load_Functions.predict_sentiment(user_input, Load_Model.xgb, Load_Model.vectorizer)
    result = Load_Functions.change_value(result1)
    st.write("Predicted Sentiment:", result)


