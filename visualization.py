from music21 import stream, meter, key, note, metadata, expressions
import music21
import json
import os

def visualize(time_signature,keys,melodies,compare=0):
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
        for n in melodies[i]:
            splitted=n.split('-')
            if len(splitted)==2:
                note,duration=splitted[0],float(splitted[1])
            else:
                note = "-".join( substring for substring in splitted[:-1])
                duration=float(splitted[-1])
            if measureTime+duration>measureDuration:
                if measureTime!=measureDuration:
                    measure.append(music21.note.Rest(quarterLength=measureDuration-(measureTime))) 
                part.append(measure)
                measure=stream.Measure(measureCount)
                measureCount+=1
                measureTime=0.0
            measureTime+=duration
            measure.append(music21.note.Note(note,quarterLength=duration))
        if measureTime!=measureDuration:
            measure.append(music21.note.Rest(quarterLength=measureDuration-(measureTime))) 
        part.append(measure)
        measureTime=0.0
        for _ in range(5):
            measure=stream.Measure(measureCount)
            measureCount+=1
            measure.append(music21.note.Rest(quarterLength=measureDuration))
            part.append(measure)
        measure=stream.Measure(measureCount)
        measureCount+=1

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



def visualize_for_all(time_signature, keys, melodies, compare=0):
    #Version légerment modifié de visualize pour l'affichage de toutes les mélodies générées
    score = stream.Score()
    score.metadata = metadata.Metadata(title="Melodies generated")
    N = len(keys)
    part = stream.Part()
    part.append(meter.TimeSignature(time_signature))
    measure = stream.Measure(0)
    measure_duration = float(time_signature.split('/')[0])
    measure_count = 1
    for i in range(N):
        if compare != 0:
            if i % 2 == 0:
                name = f"Melody {compare} generated"
            else:
                name = f"Melody {compare} Original"
        else:
            name = f"Melody {i + 1}"
        measure.append(expressions.RehearsalMark(name))
        keysplit = keys[i].split(" ")
        if keys[i] == "<music21.key.KeySignature of no sharps or flats>":
            measure.append(key.KeySignature(0))
        elif len(keysplit) == 2:
            measure.append(key.Key(keysplit[0], keysplit[1]))
        else:
            measure.append(key.Key(keys[i]))
        measure_time = 0.0
        for melody in melodies[i]:
            for n in melody:
                splitted = n.split('-')
                if len(splitted) == 2:
                    note, duration = splitted[0], float(splitted[1])
                else:
                    note = "-".join(substring for substring in splitted[:-1])
                    duration = float(splitted[-1])
                if measure_time + duration > measure_duration:
                    if measure_time != measure_duration:
                        measure.append(music21.note.Rest(quarterLength=measure_duration - (measure_time)))
                    part.append(measure)
                    measure = stream.Measure(measure_count)
                    measure_count += 1
                    measure_time = 0.0
                measure_time += duration
                measure.append(music21.note.Note(note, quarterLength=duration))
        if measure_time != measure_duration:
            measure.append(music21.note.Rest(quarterLength=measure_duration - (measure_time)))
        part.append(measure)
        measure_time = 0.0
        for _ in range(5):
            measure = stream.Measure(measure_count)
            measure_count += 1
            measure.append(music21.note.Rest(quarterLength=measure_duration))
            part.append(measure)
        measure = stream.Measure(measure_count)
        measure_count += 1

    score.append(part)
        
    return score

def show_all_generated(file_name):
    if not os.path.exists(file_name) or not os.path.isfile(file_name):
        print(f"The file '{file_name}' does not exist. Exiting program.")
        return
    
    with open(file_name, 'r') as file:
        generated_data = json.load(file)
    
    time_signature = "2/4"  # A modifier selon les besoins
    
    melodies = []
    keys = []
    for entry in generated_data:
        melodies.append([entry["Generated"], entry["Original"]])
        keys.append(entry["Key"])  # Clé sans encapsulation dans une liste
    
    score = visualize_for_all(time_signature, keys, melodies)
    score.show()


