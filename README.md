# Lyrewave
Lyrewave is an AI-Project to generate audio, based on training examples. Instead of the traditional approach to convert audio into the frequency domain, it operates on the raw audio. Lyrewave is inspired by the lyrebird, a species known for the ability to mimic sounds from it's environment.

## Setup
This package can be installed using:
```py
!pip install git+https://github.com/timowendner/lyrewave
```
Then we can import it in and run it as following:
```py
import lyrewave
lyrewave.run('path/to/config.toml')
```