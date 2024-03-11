from music21 import stream, meter, key, note, metadata, expressions, tie
import music21
import json
import os

def visualize(time_signature,keys,melodies,compare=0,tempo=80):
    """
        Convertie une liste de melodies sous forme de string "note-duree, note-duree, ..." en une partition music21 afin de pouvoir les visualiser.

        Parametres:
            time_signature (string) : Choix de la signature rythmique. Doit etre de la forme "int/int"
            keys (liste de string) : La liste des cles globales de chaque melodie. Doit etre de la meme taille que melodies.
            melodies(liste de liste de tuple(string,float)) : La liste des melodies a convertir.
        Sortie:
            score (music21 score): La partition contenant l'ensemble des melodies separees par 5 mesures de silence.
    """
    if len(keys)!=len(melodies):
        print("Il doit y avoir une cle par melodie")
        return None
    score = stream.Score()
    score.metadata = metadata.Metadata(title="Melodies generated")
    N=len(keys)
    part = stream.Part()
    part.append(meter.TimeSignature(time_signature))
    measure=stream.Measure(0)
    measureDuration=float(time_signature.split('/')[0])
    measureCount=1
    for i in range(N):
        if compare!=0 :
            if i%2==0 :
                name=f"Melody {compare} generated"
            else :
                name=f"Melody {compare} Original"
        else :
            name=f"Melody {i+1}"
        measure.append(expressions.RehearsalMark(name))
        keysplit=keys[i].split(" ")
        if keys[i]=="<music21.key.KeySignature of no sharps or flats>" :
            measure.append(key.KeySignature(0))
        elif len(keysplit)==2:
            measure.append(key.Key(keysplit[0],keysplit[1]))
        else:
            measure.append(key.Key(keys[i]))
        measureTime=0.0
        measure.append(music21.tempo.MetronomeMark(number=tempo))
        for note_duration in melodies[i]:
            note_duration=note_duration.split("-")
            if len(note_duration)==2:
                note_name,duration=note_duration[0],float(note_duration[1])
            else:
                note_name,duration="-".join(n for n in note_duration[:-1]),float(note_duration[-1])
            if measureTime==measureDuration: #si la mesure est finie on l'ajoute
                part.append(measure)
                measureTime=0.0
                measure=stream.Measure(measureCount)
                measureCount+=1

            if measureTime+duration>measureDuration: #La note dépase la mesure, on coupe
                if note_name=="rest":
                    measure.append(note.Rest(quarterLength=measureDuration-(measureTime)))
                else:
                    note1=note.Note(note_name,quarterLength=measureDuration-(measureTime))
                    measure.append(note1) 
                    note1.tie = tie.Tie()
                part.append(measure)
                measure=stream.Measure(measureCount)
                measureCount+=1
                if note_name=="rest":
                    measure.append(note.Rest(quarterLength=duration-(measureDuration-(measureTime))))
                else:
                    measure.append(note.Note(note_name,quarterLength=duration-(measureDuration-(measureTime))))
                measureTime=duration-(measureDuration-(measureTime))
            else:
                measureTime+=duration
                if note_name=="rest":
                    measure.append(note.Rest(quarterLength=duration))
                else:
                    measure.append(note.Note(note_name,quarterLength=duration))
            
        if measureTime!=measureDuration:
            measure.append(note.Rest(quarterLength=measureDuration-(measureTime))) 
        part.append(measure)
        measureTime = 0.0
        for _ in range(5):
            measure = stream.Measure(measureCount)
            measureCount += 1
            measure.append(note.Rest(quarterLength=measureDuration))
            part.append(measure)
        measure = stream.Measure(measureCount)
        measureCount += 1
    score.append(part)
        
    return score


def compare_all_generated(file_name='generated.json',rests=2):

    if not(os.path.exists(file_name) and os.path.isfile(file_name)):
        print(f"The file '{file_name}' does not exist. Exiting program.")
        return None
    
    with open(file_name, 'r') as file:
        generated = json.load(file)
    
    time_signature = "2/4"  # À modifier si nécessaire
    melodies=[]
    keys=[]
    for i in range(len(generated)):
        melodies.append(generated[i]["Generated"])
        melodies.append(generated[i]["Original"])
        keys.append(generated[i]["Key"])
        keys.append(generated[i]["Key"])
    
    score = stream.Score()
    score.metadata = metadata.Metadata(title="Melodies generated")
    N=len(keys)
    part = stream.Part()
    part.append(meter.TimeSignature(time_signature))
    measure=stream.Measure(0)
    measureDuration=float(time_signature.split('/')[0])
    measureCount=1
    for i in range(N):
        if i%2==1 :
            name=generated[i//2]["Title"]+" - "+str(generated[i//2]["Mesure"])
        else :
            name=f"Melody {(i//2)+1} generated"
        measure.append(expressions.RehearsalMark(name))
        keysplit=keys[i].split(" ")
        if keys[i]=="<music21.key.KeySignature of no sharps or flats>" :
            measure.append(key.KeySignature(0))
        elif len(keysplit)==2:
            measure.append(key.Key(keysplit[0],keysplit[1]))
        else:
            measure.append(key.Key(keys[i]))
        measureTime=0.0
        measure.append(music21.tempo.MetronomeMark(number=60))
        for note_duration in melodies[i]:
            note_duration=note_duration.split("-")
            if len(note_duration)==2:
                note_name,duration=note_duration[0],float(note_duration[1])
            else:
                note_name,duration="-".join(n for n in note_duration[:-1]),float(note_duration[-1])
            if measureTime==measureDuration: #si la mesure est finie on l'ajoute
                part.append(measure)
                measureTime=0.0
                measure=stream.Measure(measureCount)
                measureCount+=1

            if measureTime+duration>measureDuration: #La note dépase la mesure, on coupe
                if note_name=="rest":
                    measure.append(note.Rest(quarterLength=measureDuration-(measureTime)))
                else:
                    note1=note.Note(note_name,quarterLength=measureDuration-(measureTime))
                    measure.append(note1) 
                    note1.tie = tie.Tie()
                part.append(measure)
                measure=stream.Measure(measureCount)
                measureCount+=1
                if note_name=="rest":
                    measure.append(note.Rest(quarterLength=duration-(measureDuration-(measureTime))))
                else:
                    measure.append(note.Note(note_name,quarterLength=duration-(measureDuration-(measureTime))))
                measureTime=duration-(measureDuration-(measureTime))
            else:
                measureTime+=duration
                if note_name=="rest":
                    measure.append(note.Rest(quarterLength=duration))
                else:
                    measure.append(note.Note(note_name,quarterLength=duration))
            
        if measureTime!=measureDuration:
            measure.append(note.Rest(quarterLength=measureDuration-(measureTime))) 
        part.append(measure)
        measureTime = 0.0
        for _ in range(rests):
            measure = stream.Measure(measureCount)
            measureCount += 1
            measure.append(note.Rest(quarterLength=measureDuration))
            part.append(measure)
        measure = stream.Measure(measureCount)
        measureCount += 1
    score.append(part)
        
    return score

def compare_generated(i, file_name='generated.json'):
    if i <= 0:
        print("Pour sélectionner la n-ième musique générée, tapez compare_generated(n)")
        return None
    
    if not(os.path.exists(file_name) and os.path.isfile(file_name)):
        print(f"The file '{file_name}' does not exist. Exiting program.")
        return None
    
    with open(file_name, 'r') as file:
        generated = json.load(file)
    
    time_signature = "2/4"  # À modifier si nécessaire
    melodies = [generated[i - 1]["Generated"], generated[i - 1]["Original"]]
    keys = [generated[i - 1]["Key"], generated[i - 1]["Key"]]
    
    return visualize(time_signature, keys, melodies, compare=i)


def show_all_generated(file_name,tempo=60):
    if not os.path.exists(file_name) or not os.path.isfile(file_name):
        print(f"The file '{file_name}' does not exist. Exiting program.")
        return
    
    with open(file_name, 'r') as file:
        generated_data = json.load(file)
    
    time_signature = "2/4"  # A modifier selon les besoins
    
    melodies = []
    keys = []
    for entry in generated_data:
        melodies.append(entry["Generated"]) 
        keys.append(entry["Key"])  # Clé sans encapsulation dans une liste
    
    score = visualize(time_signature, keys, melodies,tempo=tempo)
    return score 

def n_measure(melody,n,time_signature):
    measureDuration = float(time_signature.split('/')[0])
    totalTime = 0.0
    for i in range(len(melody)):
        duration=float(melody[i].split("-")[-1])
        totalTime+=duration
        if totalTime==measureDuration*n:
            return melody[:i+1]
        elif totalTime>measureDuration*n:
            last_state=melody[i].split("-")
            last_state="-".join(last_state[:-1])+"-"+str(float(last_state[-1])-(totalTime-(measureDuration*n)))
            melody[i]=last_state
            return melody[:i+1]
    return melody
