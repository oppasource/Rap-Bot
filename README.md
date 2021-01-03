# Rap-Bot

Repo contains code for twitter bot [@RapBot101](https://twitter.com/rapbot101) that tweets short rap songs (or at least it tries).

The following algorithm is used (fairly simple techniques used to finish it fast :P)

- Trained a LSTM based language model from scratch using a dataset of song lyrics corpora (many are freely available).
- The trained language model is a reverse language model, given the latter word it predicts the initial words.
- Retrieved the list of trending topics from twitter.
- Used [pronouncing](https://pypi.org/project/pronouncing/) library to get rhymying words of those trending topics.
- Those words will be the last words which are passed through trained LSTM LM to get complete sentences.
- Rap is ready to be tweeted.

The repo contains code for only the Deep Learning side of the project. Twitter related code is not published, sorry I am lazy.
