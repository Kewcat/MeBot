import requests

api_key='6b78d62d953f2e279be790c92e8c92c1'
tag= 'Happy'
limit= '5'

url = f"http://ws.audioscrobbler.com/2.0/?method=tag.getTopTracks&tag={tag}&limit={limit}&api_key={api_key}&format=json"
response= requests.get(url)
data= response.json()

tracks= data['tracks']['track']
print(f"Top {limit} songs with the tag '{tag}':")

for i, track in enumerate(tracks):
    track_name = track['name']
    artist_name = track['artist']['name']
    print(f"{i+1}. {track_name} by {artist_name}")