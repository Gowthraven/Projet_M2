import json
import sys
import music21
import os
import random
from music21 import *

MARKS=['A','B','C','D']
QUARTER_DURATION= { '32nd': '0.125', '16th' : '0.25' , 'eighth' : '0.5' ,'quarter' : '1.0' ,'half' : '2.0'}


def check_folder_exists(folder_path):
    if not(os.path.exists(folder_path) and os.path.isdir(folder_path)):
        print(f"The folder '{folder_path}' does not exist. Exiting program.")
        sys.exit()

def check_file_exists(file_path):
    if not(os.path.exists(file_path) and os.path.isfile(file_path)):
        print(f"The file '{file_path}' does not exist.  Exiting program.")
        sys.exit()

def open_one_xml(folder,filename):
    check_file_exists(folder+'/'+filename)
    if filename[-3:] =="xml":
        s=music21.converter.parse(folder+"/"+filename)
        s.show("text")
        return s
    else:
        print(f'{filename} is not a xml file. Exiting program.')
        sys.exit()

def open_x_xml(folder,x):
    '''Retourne la liste des X premiers objets music21.bases.scores
    du folder contenant des fichiers XML'''
    streams=[]
    for i,file in enumerate(os.listdir(folder)[:x]) :
        if file[-3:] =="xml":
            s=music21.converter.parse(folder+"/"+file)
            streams.append(s)
            print(i+1, "fichiers XML charges"+20*" ",end='\r')
    print(100*" ") 
    print("Chargement termine")
    return streams

def data_to_json(folder):
    '''Retourne la liste des dictionnaire'''
    D=[]
    for i,file in enumerate(os.listdir(folder)) :
        if file[-3:] =="xml":
            s=music21.converter.parse(folder+"/"+file)
            dictionnary= score_to_dict(s)
            D.append(dictionnary)
    return D

def json_into_x_melody(folder,x):
    '''Enregistre la liste des x premieres melodies de data.json'''
    file_path=folder+'/'+'data.json'
    if not(os.path.exists(file_path) and os.path.isfile(file_path)):
        print(f"The file '{file_path}' does not exist.  Exiting program.")
        return []

    with open(folder+"/"+'data.json','r') as file: #le fichier crée par extract_data.py
        data= json.load(file)

    dataset = []    
    for P in data:
        parts=[]
        #derterminer les parties de la partition
        for name_part in MARKS:
            if name_part in P:
                parts.append(name_part)
        #chaque melodie
        for part in parts:
            all_notes=""
            #chaque mesure
            for k in P[part].keys():
                if k!='key':
                    notes=P[part][k]['Notes']
                    for note in notes:
                        all_notes+=note[0]+"-"+QUARTER_DURATION[note[1]]+', '
            if len(all_notes)!=0:
                dataset.append(all_notes[:-2])
                if len(dataset)==x:                #x melody
                    with open('Data/dataset.json','w') as file:
                        json.dump(dataset,file) 
                    return len(dataset)
    with open('Data/dataset.json','w') as file:
        json.dump(dataset,file) 
    return len(dataset)
            

def get_measure_indices_for_rehearsal_marks(score):
    '''Retourne les numero de mesures pour lequels la partition possede des RehearsalMark'''
    rehearsal_marks = score.recurse().getElementsByClass('RehearsalMark')
    measure_indices = []
    for mark in rehearsal_marks:
        mark_text = str(mark)
        measure = mark.getContextByClass('Measure')
        if measure:
            measure_index = measure.number
            measure_indices.append((str(mark), measure_index))
    
    return measure_indices

def get_time_signature(score):
    '''Retourne la signature du temps de la partition en str sinon une str vide'''
    time_signature = None
    for element in score.flat:
        if isinstance(element, meter.TimeSignature):
            time_signature = element
            break
    if time_signature:
        time_signature_name = time_signature.ratioString
        return time_signature_name
    else:
        return 'None'
    
def get_keys(score):
    '''Retourne la liste des clés de la partition'''
    keys=[]
    for element in score.flat:
        if isinstance(element, key.KeySignature):
            keys.append(element)
    return keys



def get_number_of_measures(score):
    '''Retourne le nombre de mesure de la partition'''
    last_measure = None
    for element in score.recurse():
        if isinstance(element, stream.Measure):
            last_measure = element

    if last_measure:
        return last_measure.number
    else:
        return 0  # Aucune mesure trouvée

def extract_notes(score):
    '''Retourne un dic { measure : liste de notes }'''
    notes_dic=dict()
    for part in score.parts:
        m = 0
        for measure in part.getElementsByClass(stream.Measure):
            m += 1
            notes = measure.getElementsByClass(music21.note.Note)
            if notes:
                notes_dic[m] = [ (n.nameWithOctave, n.duration.type) for n in notes]

    return notes_dic

def extract_chord_symbols(score):
    '''retourne un dic { num_measure : liste accord}'''
    chord_symbols_dict = {}
    for part in score.parts:
        m = 0
        for measure in part.getElementsByClass(stream.Measure):
            m += 1
            d=[] #duree locale
            ds=[] #durees globales
            chords=[]
            
            for element in measure.flat:
                if isinstance(element,harmony.ChordSymbol):
                    chords.append(element.figure)
                    if d!=[]:
                        ds.append(sum(d))
                    d=[]
                if isinstance(element,music21.note.Note) : #l'expression locale est terminee
                    d.append(float(element.duration.quarterLength))
                    
            if d!=[]:
                 ds.append(sum(d))
            if chords:
                chord_symbols_dict[m] = [ (c,d) for c,d in zip(chords,ds)]
            
            

    return chord_symbols_dict

def get_marks_repeat(score):
    '''retourne les indices des differente marque de repetition'''
    dic=dict()
    segno_mark=[]
    coda_mark=[]
    repeat_direct_left=[]
    repeat_direct_right=[]
    key_change=[]
    for part in score.parts:
        m = 0 
        for measure in part.getElementsByClass(stream.Measure):
            for element in measure.flat:
                if isinstance(element,music21.repeat.Segno):
                    segno_mark.append(measure.number)
                if isinstance(element,music21.repeat.Coda):
                    coda_mark.append(measure.number)
                if isinstance(element,music21.key.KeySignature):
                    key_change.append(measure.number)
                if isinstance(element, music21.bar.Repeat):
                    if element.direction=='start':
                        repeat_direct_left.append(measure.number)
                    else:
                        repeat_direct_right.append(measure.number)
    dic['Segno']=segno_mark
    dic['Coda']=coda_mark
    dic['Start']=repeat_direct_left
    dic['End']=repeat_direct_right
    dic['Key']=key_change
    
    return dic
    
def extract_expression(score):
    '''retourne un dic { num_measure : liste expression (accord)}'''
    exp_dict = {}
    for part in score.parts:
        m = 0
        for measure in part.getElementsByClass(stream.Measure):
            m+= 1
            e=[] #expression locale
            d=[] #duree locale
            exp=[] #expressions de la mesure
            ds=[] #durees globales
            for element in measure.flat:
                if isinstance(element,expressions.TextExpression):
                    e.append(element.content)
                    if d!=[]:
                        ds.append(sum(d))
                    d=[]
                if isinstance(element,music21.note.Note) and e!=[]: #l'expression locale est terminee
                    exp.append(''.join(e))
                    d.append(float(element.duration.quarterLength))
                    
            if d!=[]:
                 ds.append(sum(d))
            if e!=[]:
                exp.append(''.join(e))
            if exp!=[]:
                exp_dict[m] = [ (e,d) for e,d in zip(exp,ds)]
    return exp_dict

def get_repeat(score):
    '''Retourne les numero de mesures pour lequels la partition possede des RehearsalMark'''
    repeat_mark = score.recurse().getElementsByClass('RepeatBracket')
    repeat_dict=dict()
    
    r=0

    for mark in repeat_mark:
        measures=mark.getSpannedElements()
        if mark.number=='1' or mark.number=='':
            repeats=[]
            r+=1
            repeat_dict[r]=[]
        if len(measures)==1:
            repeat_dict[r]+=[( measures[0].number,measures[0].number)]
        else:
            repeat_dict[r]+=[( measures[0].number,measures[1].number)]
    return repeat_dict

def show_stat(score):
    nb_measures= get_number_of_measures(score)
    chord_symbols = extract_chord_symbols(score)
    note_symbols=extract_notes(score)
    exp_symbols= extract_expression(score)
    rehearsal_mark_indices = get_measure_indices_for_rehearsal_marks(score)
    repeat_mark=get_repeat(score)
    marks=get_marks_repeat(score)
    keys=get_keys(score)
    t=get_time_signature(score)


    print("Titre : ",score.metadata.title)
    print('Repeats : ',  repeat_mark)
    print('Marques: ',  marks)
    

    print('keys: ',keys)
    print('time signature: ',t)
    print('marques de repet :',rehearsal_mark_indices)
    for m in range(1,nb_measures+1):
        print(f"Mesure {m}:")
        if m in chord_symbols:
            print('Accord:',chord_symbols[m])
           
        if m in exp_symbols:
            print('Expression:',exp_symbols[m])

        if m in note_symbols:
            print('Notes:',note_symbols[m])
            
        print('-----')

def score_to_dict(score):
    D=dict()
    nb_measures= get_number_of_measures(score)
    chord_symbols = extract_chord_symbols(score)
    note_symbols=extract_notes(score)
    exp_symbols= extract_expression(score)
    rehearsal_mark_indices = get_measure_indices_for_rehearsal_marks(score)
    repeat_mark=get_repeat(score)
    marks=get_marks_repeat(score)
    keys=get_keys(score)
    t=get_time_signature(score)
    starts= [ i for m,i in rehearsal_mark_indices] +[nb_measures+1]
    segments_end = [ s[0][1] for s in repeat_mark.values()]
    D['title']=score.metadata.title
    D['time_signature']=t
    seg_i=0
    for mark,(_,i) in enumerate(rehearsal_mark_indices):
        D[MARKS[mark]]= dict()
        j= i #start
        m_i=1
        D[MARKS[mark]]["key"]= str(keys[mark]) if mark< len(keys) else str(keys[0])
        while j < starts[mark+1]:
            D[MARKS[mark]][m_i]=dict()
            D[MARKS[mark]][m_i]["Notes"] = note_symbols[i] if i in note_symbols else []
            D[MARKS[mark]][m_i]["Expressions"] = exp_symbols[i] if i in exp_symbols else []
            D[MARKS[mark]][m_i]["Accords"] = chord_symbols[i] if i in chord_symbols else []
            m_i+=1
            j+=1
            if seg_i < len(segments_end): # il y a toujours des segments
                if j-1== segments_end[seg_i] and segments_end[seg_i]<starts[mark+1]: #segment correspond a la partie
                    seg_i+=1
                    break
    return D

def extract_random_seq(jsonfile,lg):
    '''Retourne (nom de la partition, nom de la partie, clé, et les lg premieres notes sous forme de string) d'une partie d'une partition au hasard'''
    if not(os.path.isfile(jsonfile)):
        print(f"The file '{jsonfile}' does not exist.  Exiting program.")
        return ()

    with open(jsonfile,'r') as file: #le fichier crée par extract_data.py
        data = json.load(file)

    selected_notes = []

    while len(selected_notes) < lg: # au cas où il n'y aurait pas assez de notes dans la partie de la partition sélectionnée

        # choix au hasard d'une partition
        selected_score = random.choice(data)

        # choix au hasard d'une partie de cette partition
        parts = [key for key in selected_score.keys() if key in MARKS]
        name_part = random.choice(parts)
        selected_part = selected_score[name_part]

        # conservation des lg premières notes
        all_notes = [item['Notes'] for measure, item in selected_part.items() if measure.isdigit()]
        selected_notes = [item for sublist in all_notes for item in sublist][:lg]
        selected_notes= [str(n)+"-"+str(QUARTER_DURATION[d]) for n,d in selected_notes]

    random_seq = [selected_score['title'],name_part,selected_part['key'],selected_notes]

    return random_seq

def extract_seq_from(jsonfile,desired_score,desired_part):
    '''Retourne (nom de la partition, nom de la partie, clé, et les lg premieres notes sous forme de string) de la partie d'une partition données'''
    if not(os.path.isfile(jsonfile)):
        print(f"The file '{jsonfile}' does not exist.  Exiting program.")
        return ()

    with open(jsonfile,'r') as file: #le fichier crée par extract_data.py
        data = json.load(file)

    selected_score = None
    selected_part = None

    for score in data:
        if score.get('title') == desired_score:
            selected_score = score
            break

    if selected_score == None:
        print("La partition",desired_score,"n'existe pas.")
        return ()

    for part in selected_score.keys():
        if part == desired_part:
            selected_part = selected_score[part]
            all_notes = [item['Notes'] for measure, item in selected_part.items() if measure.isdigit()]
            all_notes= [str(n)+"-"+str(QUARTER_DURATION[d]) for notes in all_notes for n,d in notes]
            break

    if selected_part == None:
            print("La partie",desired_part,"de la partition",desired_score,"n'existe pas.")
            return ()

    seq_from = [selected_score['title'],part,selected_part['key'],all_notes]

    return seq_from

if __name__ == "__main__":
    path="data/data_xml"
    check_folder_exists(path)
    if len(sys.argv) == 2:
        score_name=sys.argv[1]
        if score_name =="melodies":
            json_into_x_melody("data/",-1)
            print("Toutes les melodies ont étés générées.")
        else:
            s=open_one_xml(path,score_name)
            show_stat(s)

    elif len(sys.argv) == 1:
        file_output="data/data.json"
        D= data_to_json(path)
        with open(file_output, 'w',encoding="utf-8") as f:
            json.dump(D,f,indent=2)
        print(f'{file_output} with {len(D)} scores created.')

    elif len(sys.argv) == 3:
        x=int(sys.argv[1])
        y=sys.argv[2]
        if y =="melodies":
            n=json_into_x_melody("data/",x)
            print(f'{n} melodies generated')
        elif y == "random":
            jsonfile = "data/data.json"
            seq= extract_random_seq(jsonfile,x)
            print(f'Random sequence of {x} notes generated {seq}.')
        else:
            print("Le deuxieme argument != melodies ou != random")
            sys.exit()

    elif len(sys.argv) == 4:
        score = sys.argv[1]
        part = sys.argv[2]
        x = sys.argv[3]
        if x == "random":
            jsonfile = "data/data.json"
            seq=extract_seq_from(jsonfile,score,part)
            print(f'Sequence generated {seq}.')
        else:
            print("Le troisième argument != random")
            sys.exit()

    else:
        print("L'application prend soit 1 fichier xml en entree (affichage de la partition) soit 0 argument (tous les XML sont extraits dans data.jspn")
        sys.exit()
