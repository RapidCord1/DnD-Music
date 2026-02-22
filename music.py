import numpy as np
import simpleaudio as sa

SR = 44100

NOTE_BASE = {
 'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11
}

def note_to_freq(note):
    if note.lower() == "rest":
        return None
    name = note[:-1]
    octave = int(note[-1])
    semis_from_A4 = NOTE_BASE[name] + (octave - 4) * 12 - 9
    return 440.0 * (2.0 ** (semis_from_A4 / 12.0))

def adsr(length_s, a=0.01, d=0.05, s=0.9, r=0.06):
    n = max(1, int(length_s*SR))
    env = np.ones(n)
    a_n = int(min(n,max(1,int(a*SR))))
    d_n = int(min(n-a_n,int(d*SR)))
    r_n = int(min(n-a_n-d_n,int(r*SR)))
    sustain_n = n-(a_n+d_n+r_n)
    if a_n>0: env[:a_n] = np.linspace(0,1,a_n)
    if d_n>0: env[a_n:a_n+d_n] = np.linspace(1,s,d_n)
    if sustain_n>0: env[a_n+d_n:a_n+d_n+sustain_n]=s
    if r_n>0: env[-r_n:] = np.linspace(s,0,r_n)
    return env

def flute_tone(freq,dur,vol=0.4):
    t = np.linspace(0,dur,int(SR*dur),False)
    wave = np.sin(2*np.pi*freq*t)
    env = adsr(dur)
    return wave*env*vol

def clarinet_tone(freq,dur,vol=0.35):
    t = np.linspace(0,dur,int(SR*dur),False)
    wave = 0.9*np.sin(2*np.pi*freq*t)+0.2*np.sin(2*np.pi*3*freq*t)
    env = adsr(dur)
    return wave*env*vol

def alto_tone(freq,dur,vol=0.32):
    t = np.linspace(0,dur,int(SR*dur),False)
    wave = 0.6*np.sin(2*np.pi*freq*t)+0.18*np.sin(2*np.pi*2*freq*t)
    env = adsr(dur)
    return wave*env*vol

def trumpet_tone(freq,dur,vol=0.3):
    t = np.linspace(0,dur,int(SR*dur),False)
    wave = 0.7*np.sin(2*np.pi*freq*t)+0.35*np.sin(2*np.pi*2*freq*t)
    env = adsr(dur)
    return wave*env*vol

def bass_drum_thump(dur,vol=0.2):
    t = np.linspace(0,dur,int(SR*dur),False)
    wave = np.sin(2*np.pi*60*t)*np.exp(-6*t)
    wave += 0.02*np.random.normal(size=wave.shape)
    return wave*vol

def snare_soft(dur,vol=0.1):
    t = np.linspace(0,dur,int(SR*dur),False)
    wave = np.random.normal(0,1,len(t))
    wave = np.convolve(wave,np.ones(32)/32,mode='same')
    env = np.linspace(1,0,len(wave))
    return wave*env*vol

# Tempo
bpm = 80
q_sec = 60/bpm
measures = 4
measure_len = q_sec*4
total_time = measure_len*measures + 2.5
track = np.zeros(int(total_time*SR))

# Score
flute_events = [
    ("F4",0.0,1*q_sec),("G4",1*q_sec,0.5*q_sec),("F4",1.5*q_sec,0.5*q_sec),
    ("Ab4",2*q_sec,1*q_sec),("rest",3*q_sec,1*q_sec),
    ("F4",4*q_sec,1*q_sec),("G4",5*q_sec,0.5*q_sec),("F4",5.5*q_sec,0.5*q_sec),
    ("Bb4",6*q_sec,1*q_sec),("rest",7*q_sec,1*q_sec)
]

clarinet_events = [
    ("D4",0.0,1*q_sec),("rest",1*q_sec,1*q_sec),("E4",2*q_sec,1*q_sec),("D4",3*q_sec,1*q_sec),
    ("E4",4*q_sec,0.5*q_sec),("D4",4.5*q_sec,0.5*q_sec),("G4",5*q_sec,1*q_sec),("D4",6*q_sec,1*q_sec),
    ("F4",7*q_sec,1*q_sec)
]

alto_events = [
    ("F4",0.0,2*q_sec),("G4",2*q_sec,2*q_sec),
    ("E4",4*q_sec,2*q_sec),("F4",6*q_sec,2*q_sec)
]

trumpet_events = [
    ("F4",1.0*q_sec,1*q_sec),("C5",2.0*q_sec,1*q_sec),("Bb4",3.0*q_sec,1*q_sec)
]

bd_events = [
    (0.0,0.0,0.12), (1*q_sec,0.0,0.12), (2*q_sec,0.0,0.12), (3*q_sec,0.0,0.12)
]

sn_events = [
    (0.0,2*q_sec,0.12),(1*q_sec,2*q_sec,0.12),(2*q_sec,2*q_sec,0.12),(3*q_sec,2*q_sec,0.12)
]

def place_event(track,instr,note,start,dur,vol=1.0):
    if note.lower() == "rest":
        return track
    si = int(start*SR)
    ei = si + int(dur*SR)
    if instr=="flute": wave = flute_tone(note_to_freq(note),dur,vol)
    elif instr=="clar": wave = clarinet_tone(note_to_freq(note),dur,vol)
    elif instr=="alto": wave = alto_tone(note_to_freq(note),dur,vol)
    elif instr=="trump": wave = trumpet_tone(note_to_freq(note),dur,vol)
    elif instr=="bd": wave = bass_drum_thump(dur,vol)
    elif instr=="sn": wave = snare_soft(dur,vol)
    else: return track
    if ei>len(track): track.resize(ei+1000,refcheck=False)
    track[si:ei]+=wave
    return track

# Place Events
for n,s,d in flute_events: track=place_event(track,'flute',n,s,d,0.36)
for n,s,d in clarinet_events: track=place_event(track,'clar',n,s,d,0.30)
for n,s,d in alto_events: track=place_event(track,'alto',n,s,d,0.28)
for n,s,d in trumpet_events: track=place_event(track,'trump',n,s,d,0.26)
for s,o,d in bd_events: track=place_event(track,'bd',"F2",s,d,0.18)
for s,o,d in sn_events: track=place_event(track,'sn',"snare",s,d,0.08)

# Boom Chord (measure 4)
boom_notes = ["F5","Ab5","C6","F5"]
for i,note in enumerate(boom_notes):
    wave = trumpet_tone(note_to_freq(note),1.6,0.6)
    start = 3*q_sec + 0.0
    si = int(start*SR)+i*50
    ei = si+len(wave)
    if ei>len(track): track.resize(ei+1000,refcheck=False)
    track[si:ei]+=wave

bd_big = bass_drum_thump(1.0,0.9)
si = int(3*q_sec*SR)
ei = si+len(bd_big)
track[si:ei]+=bd_big

# Normalize and play
track /= np.max(np.abs(track))
audio = (track*32767).astype(np.int16)
print("Playing 4-measure sneaky -> high boom...")
sa.play_buffer(audio,1,2,SR).wait_done()
print("Song Playing")
