import torchaudio
import matplotlib.pyplot as plt 
# from pydub import AudioSegment

# Example usage
if __name__ == "__main__":
    # data 
    file1 = "data/clips/common_voice_en_41236242.mp3"
    track_1, sample_rate = torchaudio.load(str(file1), format="mp3")
    print("Sample rate:",sample_rate)
    print("Track shape:",track_1.shape)
    plt.plot(track_1[0][55000:70000])
    plt.show()
    torchaudio.save('data/output/output.wav', track_1, sample_rate)
    # AudioSegment.from_wav("data/output/output.wav").export("data/output/output.mp3", format="mp3")
