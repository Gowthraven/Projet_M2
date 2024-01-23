from music21 import stream, meter, key, note, metadata, expressions
import music21

def visualize(time_signature,keys,melodies):
    score = stream.Score()
    score.metadata = metadata.Metadata(title="Melodies generated")
    N=len(keys)
    part = stream.Part()
    part.append(meter.TimeSignature(time_signature))
    measure=stream.Measure(0)
    measureDuration=float(time_signature.split('/')[0])
    measureCount=1
    for i in range(N):
        measure.append(expressions.RehearsalMark(f"Melody {i+1}")) 
        measure.append(key.Key(keys[i]))
        measureTime=0.0
        for n in melodies[i].split(', '):
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