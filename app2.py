import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from textblob import TextBlob
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

#title
st.title('Sentiment Analysis Tool')
st.sidebar.title('Airlines Data')

data=pd.read_csv('tweet.csv')
if st.checkbox("Show Data"):
    st.write(data.head(100))
#subheader
st.sidebar.subheader('Select the sentiment to see the raw data')
#radio buttons
tweets=st.sidebar.radio('Sentiment Type',('positive','negative','neutral'))
st.write(data.query('airline_sentiment==@tweets')[['text']].sample(1).iat[0,0])
st.write(data.query('airline_sentiment==@tweets')[['text']].sample(1).iat[0,0])
st.write(data.query('airline_sentiment==@tweets')[['text']].sample(1).iat[0,0])
#selectbox + visualisation
# An optional string to use as the unique key for the widget. If this is omitted, a key will be generated for the widget based on its content.
## Multiple widgets of the same type may not share the same key.
select=st.sidebar.selectbox('Visualisation Of Tweets',['Histogram','Pie Chart'],key=1)
sentiment=data['airline_sentiment'].value_counts()
sentiment=pd.DataFrame({'Sentiment':sentiment.index,'Tweets':sentiment.values})
st.markdown("###  Sentiment count")
if select == "Histogram":
        fig = px.bar(sentiment, x='Sentiment', y='Tweets', color = 'Tweets', height= 500)
        st.plotly_chart(fig)
else:
        fig = px.pie(sentiment, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

#slider
st.sidebar.markdown('Time & Location of tweets')
hr = st.sidebar.slider("Hour of the day", 0, 23)
data['Date'] = pd.to_datetime(data['tweet_created'])
hr_data = data[data['Date'].dt.hour == hr]
if not st.sidebar.checkbox("Hide", True, key='1'):
    st.markdown("### Location of the tweets based on the hour of the day")
    st.markdown("%i tweets during  %i:00 and %i:00" % (len(hr_data), hr, (hr+1)%24))
    st.map(hr_data)

#multiselect
st.sidebar.subheader("Airline tweets by sentiment")
choice = st.sidebar.multiselect("Airlines", ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'), key = '0')  
if len(choice)>0:
    air_data=data[data.airline.isin(choice)]
    # facet_col = 'airline_sentiment'
    fig1 = px.histogram(air_data, x='airline', y='airline_sentiment', histfunc='count', color='airline_sentiment',labels={'airline_sentiment':'tweets'}, height=600, width=800)
    st.plotly_chart(fig1)


# sentiment check
# Fxn
def convert_to_df(sentiment):
	sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
	return sentiment_df

def analyze_token_sentiment(docx):
	analyzer = SentimentIntensityAnalyzer()
	pos_list = []
	neg_list = []
	neu_list = []
	for i in docx.split():
		res = analyzer.polarity_scores(i)['compound']
		if res > 0.1:
			pos_list.append(i)
			pos_list.append(res)

		elif res <= -0.1:
			neg_list.append(i)
			neg_list.append(res)
		else:
			neu_list.append(i)

	result = {'positives':pos_list,'negatives':neg_list,'neutral':neu_list}
	return result 


st.title("Check the sentiment of custom data")
with st.form(key='nlpForm'):
	raw_text = st.text_area("Enter Text Here")
	submit_button = st.form_submit_button(label='Analyze')

	# layout
	col1,col2 = st.columns(2)
	if submit_button:

		with col1:
			st.info("Results")
			sentiment = TextBlob(raw_text).sentiment
			st.write(sentiment)

			# Emoji
			if sentiment.polarity > 0:
				st.markdown("Sentiment:: Positive :smiley: ")
			elif sentiment.polarity < 0:
				st.markdown("Sentiment:: Negative :angry: ")
			else:
				st.markdown("Sentiment:: Neutral 😐 ")

			# Dataframe
			result_df = convert_to_df(sentiment)
			st.dataframe(result_df)

			# Visualization
			c = alt.Chart(result_df).mark_bar().encode(
				x='metric',
				y='value',
				color='metric')
			st.altair_chart(c,use_container_width=True)



		with col2:
			st.info("Token Sentiment")

			token_sentiments = analyze_token_sentiment(raw_text)
			st.write(token_sentiments)