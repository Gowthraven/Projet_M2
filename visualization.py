from music21 import stream, meter, key, note, metadata, expressions
import music21

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
        if len(keysplit)==2:
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

def compare_generated(i):
    if i<=0 :
        print("Pour selectionnez la n-ième musique générée tapez compare_generated(n)")
    file_path='generated.json'
    
    if not(os.path.exists(file_path) and os.path.isfile(file_path)):
            print(f"The file '{file_path}' does not exist.  Exiting program.")
            return None
    with open(file_path,'r') as file: #le fichier crée par extract_data.py
            generated= json.load(file)
            
    time_signature="2/4"                     #à changer !!!
    melodies=[generated[i-1]["Generated"],generated[i-1]["Original"]]
    keys=[generated[i-1]["Key"],generated[i-1]["Key"]]
    
    return visualize(time_signature,keys,melodies,compare=i)