import nltk
# tag explanation: https://www.geeksforgeeks.org/part-speech-tagging-stop-words-using-nltk-python/
# tree docs: https://www.nltk.org/_modules/nltk/tree.html
# more tree docs: http://www.nltk.org/howto/tree.html
# grammar docs: https://www.nltk.org/book/ch08.html and http://www.nltk.org/howto/generate.html
# from nltk.draw.util import CanvasFrame
# from nltk.draw import TreeWidget

def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i][1] == pattern[0] and mylist[i:i+len(pattern)][1] == pattern:
            matches.append([i,i+len(pattern)])
    return matches

def pattern_finder(tags_curr, pattern, pattern_name):
    '''modifies and returns tags list'''
    tags, locs = tags_curr
    for i in range(len(tags)):
        if tags[i:i+len(pattern)] == pattern:
            tags = tags[:i] + [pattern_name] + tags[i+len(pattern):]
            locs.append()
    return (tags, locs)

# def semantic_parser(sentence):
#     print('semantic parser received sentence: ', sentence)
#     tokens = nltk.word_tokenize(sentence)
#     tagged = nltk.pos_tag(tokens)
#     # nltk doesn't seem to correctly identify some verbs
#     keywords = {'put': 'VB', 'place': 'VB', 'pour': 'VB', 'pick': 'VB'}
#     for i,(word,tag) in enumerate(tagged):
#         if word in keywords.keys():
#             tagged[i] = (word, keywords[word])
#     tags = [tag for (word, tag) in tagged]
#     print(tagged)

#     # look for the preposition
#     if 'IN' in tags and tokens[0] in ["put", "place"]:
#         prep_idx = tags.index('IN')
#         VB = tagged[:prep_idx]
#         PP = tagged[prep_idx:]

#         # pick up task
#         # look for the form "DT JJ NN", e.g. "the green cup"
#         st = "pick up the"
#         jj = ""
#         if 'JJ' in tags[:prep_idx]:
#             jj = tokens[tags[:prep_idx].index('JJ')]
#         nn = tokens[tags[:prep_idx].index('NN')]
#         pick_up_task = ' '.join([st, jj, nn])

#         # put down task
#         # st = "put it in the"
#         st = "pour all of it into the"
#         jj = ""
#         if 'JJ' in tags[prep_idx:]:
#             jj = tokens[tags[prep_idx:].index('JJ')+prep_idx]
#         # nn = tokens[tags[prep_idx:].index('NN')+prep_idx]
#         nn = "dish"
#         put_down_task = ' '.join([st, jj, nn])

#         print('generated subtasks: ')
#         print([pick_up_task, put_down_task])
#         return [pick_up_task, put_down_task]

#     print('no subtasks generated: ')
#     print([sentence])
#     return [sentence]
#     # for (word, tag) in tagged:
#     #     print((word, tag))

num_to_int = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6}

def semantic_parser(sentence):
    # manually override tensor input at initialization
    if type(sentence) != str:
        return []
    # pattern matching
    print('\n------------\n')
    print('semantic parser received:', sentence)
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    # nltk doesn't seem to correctly identify some verbs
    keywords = {'put': 'VB', 'place': 'VB', 'pour': 'VB', 'pick': 'VB'}
    for i,(word,tag) in enumerate(tagged):
        if word in keywords.keys():
            tagged[i] = (word, keywords[word])    
    tags = [tag for (word, tag) in tagged]
    # print(tagged)

    grammar_string = """
        S -> V O | V O L | V O CC V O L | V O CC O L | V O CC O CC O L
        V -> 'VB' | 'VB' 'RP'
        O -> 'DT' 'JJ' 'NN' | 'DT' 'NN' | 'PDT' 'DT' 'JJ' 'NNS' | 'PDT' 'DT' 'NNS' | 'PDT' 'JJ' 'NNS' | 'PDT' 'NNS' | 'CD' 'JJ' 'NN' | 'CD' 'JJ' 'NNS' | 'CD' 'NN' | 'CD' 'NNS' | 'PRP'
        L -> 'IN' 'DT' 'NN' | 'IN' 'DT' 'JJ' 'NN'
        CC -> 'CC'
    """
    def get_trees(tags):
        trees = []
        grammar = nltk.CFG.fromstring(grammar_string)
        parser = nltk.ChartParser(grammar)
        for tree in parser.parse(tags):
            # cf = CanvasFrame()
            trees.append(tree)
            # tc = TreeWidget(cf.canvas(),tree)
            # cf.add_widget(tc,10,10) # (10,10) offsets
            # cf.print_to_file('tree.ps')
            # tree.draw()
            # cf.destroy()
        return trees
    
    def get_abs_idx(tree, tree_idx):
        rel_idx = [len(i) for i in tree]
        return sum(rel_idx[:tree_idx[0]]) + (tree_idx[1]%len(tree[tree_idx[0]]))
        # for idx in rel_idx:
        #     remainder = abs_idx - idx
    
    def generate_pick_tasks(tree, subtree_idx):
        subtasks = []
        obj_l = tokens[get_abs_idx(tree,[subtree_idx,0]):get_abs_idx(tree,[subtree_idx,0])+len(tree[subtree_idx])]
        if tree[subtree_idx,-1] == 'NN':
            # single object
            subtask = ' '.join(['pick', 'up'] + obj_l)
            subtasks.append(subtask)
        elif tree[subtree_idx,-1] == 'NNS':
            # multiple objects
            pass #TODO
            if tree[subtree_idx, 0] == 'CD':
                count = num_to_int[tokens[get_abs_idx(tree, [subtree_idx,0])].lower()]
                for i in range(count):
                    subtask = ' '.join(['pick', 'up', 'the'] + obj_l[1:-1] + ['cup'])
                    subtasks.append(subtask)
        return subtasks
    
    def generate_place_tasks(tree, subtree_idx):
        subtasks = []
        loc_l = tokens[get_abs_idx(tree,[subtree_idx,0]):get_abs_idx(tree,[subtree_idx,0])+len(tree[subtree_idx])]
        # subtask = ' '.join(['place', 'it'] + loc_l)
        # TODO change once "place" is recognized
        subtask = ' '.join(['place', 'it'] + loc_l)
        subtasks.append(subtask)
        return subtasks
    
    trees = get_trees(tags)
    if len(trees) > 1:
        print('WARNING: more than one tree generated')

    for tree in trees:
        tree.pretty_print()
        if tree.label() != 'S':
            print('ERROR: sentence is not correctly structured')
            return []
        if len(tree) not in [2,3,5,6,7]:
            print('ERROR: grammar appears to be misformed')
            return []
        if len(tree) == 2:
            # V O
            print('no subtasks generated')
            print([sentence])
            return [sentence]
        elif len(tree) == 3:
            # V O L
            subtasks = []
            # # pick task
            # if tree[1,-1] == 'NN':
            #     # single object
            #     obj_l = tokens[get_abs_idx(tree,[1,0]):get_abs_idx(tree,[1,0])+len(tree[1])]
            #     subtask = ' '.join(['pick', 'up'] + obj_l)
            #     subtasks.append(subtask)
            # elif tree[1,-1] == 'NNS':
            #     # multiple objects
            #     pass #TODO
            subtasks += generate_pick_tasks(tree, 1)
            # loc_l = tokens[get_abs_idx(tree,[2,0]):get_abs_idx(tree,[2,0])+len(tree[2])]
            # subtask = ' '.join(['place', 'it'] + loc_l)
            # subtasks.append(subtask)
            place_subtasks = generate_place_tasks(tree, 2)*len(subtasks)
            # interleave
            subtasks = [val for pair in zip(subtasks, place_subtasks) for val in pair]
            print('generated subtasks:', subtasks)
            return subtasks
        elif len(tree) == 5:
            # V O CC O L
            subtasks = []
            # if tree[1,-1] == 'NN':
            #     # single object
            #     obj_l = tokens[get_abs_idx(tree,[1,0]):get_abs_idx(tree,[1,0])+len(tree[1])]
            #     subtask = ' '.join(['pick', 'up'] + obj_l)
            #     subtasks.append(subtask)
            # elif tree[1,-1] == 'NNS':
            #     # multiple objects
            #     pass #TODO
            subtasks += generate_pick_tasks(tree, 1)
            # if tree[3,-1] == 'NN':
            #     # single object
            #     obj_l = tokens[get_abs_idx(tree,[3,0]):get_abs_idx(tree,[3,0])+len(tree[3])]
            #     subtask = ' '.join(['pick', 'up'] + obj_l)
            #     subtasks.append(subtask)
            # elif tree[3,-1] == 'NNS':
            #     # multiple objects
            #     pass #TODO
            subtasks += generate_pick_tasks(tree, 3)
            # loc_l = tokens[get_abs_idx(tree,[4,0]):get_abs_idx(tree,[4,0])+len(tree[4])]
            # subtask = ' '.join(['place', 'it'] + loc_l)
            # subtasks.append(subtask)
            place_subtasks = generate_place_tasks(tree, 4)*len(subtasks)
            # interleave
            subtasks = [val for pair in zip(subtasks, place_subtasks) for val in pair]
            print('generated subtasks:', subtasks)
            return subtasks
        elif len(tree) == 6:
            # V O CC V O L
            subtasks = []
            # if tree[1,-1] == 'NN':
            #     # single object
            #     obj_l = tokens[get_abs_idx(tree,[1,0]):get_abs_idx(tree,[1,0])+len(tree[1])]
            #     subtask = ' '.join(['pick', 'up'] + obj_l)
            #     subtasks.append(subtask)
            # elif tree[1,-1] == 'NNS':
            #     # multiple objects
            #     pass #TODO
            subtasks += generate_pick_tasks(tree, 1)
            # loc_l = tokens[get_abs_idx(tree,[5,0]):get_abs_idx(tree,[5,0])+len(tree[5])]
            # subtask = ' '.join(['place', 'it'] + loc_l)
            # subtasks.append(subtask)
            place_subtasks = generate_place_tasks(tree, 5)*len(subtasks)
            # interleave
            subtasks = [val for pair in zip(subtasks, place_subtasks) for val in pair]
            print('generated subtasks:', subtasks)
            return subtasks
        elif len(tree) == 7:
            # V O CC O CC O L
            return [] # TODO
        
        

def _parser_old(sentence):
    action = None
    # pattern matching
    print('semantic parser received sentence: ', sentence)
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    # nltk doesn't seem to correctly identify some verbs
    keywords = {'put': 'VB', 'place': 'VB', 'pour': 'VB', 'pick': 'VB'}
    for i,(word,tag) in enumerate(tagged):
        if word in keywords.keys():
            tagged[i] = (word, keywords[word])    
    tags = [tag for (word, tag) in tagged]
    print(tagged)

    # step 1: segment sentence based on POS tags
    segments = {'V': [['VB', 'RP'], ['VB']],
    'L': [['IN', 'DT', 'JJ', 'NN'], ['IN', 'DT', 'NN']],
    'O': [['DT', 'JJ', 'NN'], ['DT', 'NN'], ['PDT', 'DT', 'JJ', 'NNS'], ['PDT', 'DT', 'NNS'], ['PDT', 'JJ', 'NNS'], ['PDT', 'NNS'], ['CD', 'JJ', 'NN'], ['CD', 'JJ', 'NNS'], ['CD', 'NN'], ['CD', 'NNS'], ['PRP']]}
    segmented = tags[:]
    for pattern_name, patterns in segments.items():
        for pattern in patterns:
            while pattern_finder(segmented, pattern, pattern_name) != segmented:
                segmented = pattern_finder(segmented, pattern, pattern_name)
    print(segmented)

    # step 2: identify action
    templates = {'pick': [['V','O']],
                 'place': [['V','O','L'],['V','O','CC','O','L'],['V','O','CC','O','CC','O','L']],
                 'pick_and_place': [['V','O','CC','V','O','L']]}
    
    for action_name, actions in templates.items():
        for template in actions:
            if segmented == template:
                action = action_name
                print('action:', action_name)
                print(segmented)
                print(sentence)
    
    if action is None:
        print('ERROR: sentence is not correctly structured')
        return []
    
    # step 3: generate subtask commands
    if action == 'pick':
        print('no subtasks generated: ')
        print([sentence])
        return [sentence]
    else:
        loc = tokens[tags.index('IN'):] # ['in','the','bin']
        objs = tokens[tags.index('')]
        subtasks = []
        



    # outputs:
    # "pick up the red cup" [VB RP DT JJ NN]: V(PICK_UP) O_0
    # "place it in the red bin" [VB PRP IN DT JJ NN]: V(PLACE) O_0/PRP L




if __name__ == "__main__":
    sentences = [
        # "pick up the cup",
        # "put it in the bin",
        # "put the green cup in the red bin",
        # "pick up the cup and put it in the bin",
        # "put all the yellow cups in the red bin",
        "place two red cups and one green cup in the red bin"
        # "pick up the red cup and put it in the bin"
    ]
    for sentence in sentences:
        semantic_parser(sentence)