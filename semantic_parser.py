import nltk
# tag explanation: https://www.geeksforgeeks.org/part-speech-tagging-stop-words-using-nltk-python/
# tree docs: https://www.nltk.org/_modules/nltk/tree.html
# more tree docs: http://www.nltk.org/howto/tree.html
# grammar docs: https://www.nltk.org/book/ch08.html and http://www.nltk.org/howto/generate.html

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

num_to_int = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6}
cup_colormap = {'red':21, 'green':22, 'blue':23}

def semantic_parser(sentence, feature_ids=[]):
    # manually override tensor input at initialization
    if type(sentence) != str:
        return []
    # pattern matching
    print('\n------------\n')
    sentence = sentence.lower()
    print('semantic parser received:', sentence)
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    # nltk doesn't seem to correctly tag some words
    keywords = {'put': 'VB', 'place': 'VB', 'pour': 'VB', 'pick': 'VB',
                'all': 'PDT',
                'cup': 'NN', 'cups': 'NNS',
                'red': 'JJ', 'yellow': 'JJ', 'green': 'JJ', 'blue': 'JJ', 'pink': 'JJ'}
    for i,(word,tag) in enumerate(tagged):
        if word in keywords.keys():
            tagged[i] = (word, keywords[word])    
    tags = [tag for (word, tag) in tagged]

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
            trees.append(tree)
        return trees
    
    def get_abs_idx(tree, tree_idx):
        rel_idx = [len(i) for i in tree]
        return sum(rel_idx[:tree_idx[0]]) + (tree_idx[1]%len(tree[tree_idx[0]]))
    
    def generate_pick_tasks(tree, subtree_idx):
        subtasks = []
        obj_l = tokens[get_abs_idx(tree,[subtree_idx,0]):get_abs_idx(tree,[subtree_idx,0])+len(tree[subtree_idx])]
        if tree[subtree_idx,-1] == 'NN':
            # single object
            subtask = ' '.join(['pick', 'up'] + obj_l)
            subtasks.append(subtask)
        elif tree[subtree_idx,-1] == 'NNS':
            # multiple objects
            if tree[subtree_idx, 0] == 'PDT':
                # "all" command. how many objects?
                # if color specified in object, count that
                if 'JJ' in tree[subtree_idx]:
                    color_idx = tree[subtree_idx].index('JJ')
                    color = tokens[get_abs_idx(tree,[subtree_idx,color_idx])]
                    obj_idx = cup_colormap[color]
                    count = feature_ids.count(obj_idx)
                    for i in range(count):
                        if obj_l[1] == 'the':
                            subtask = ' '.join(['pick', 'up'] + obj_l[1:-1] + ['cup'])
                        else:
                            subtask = ' '.join(['pick', 'up', 'the'] + obj_l[1:-1] + ['cup'])
                        subtasks.append(subtask)
                # if color not specified, all the cups in the scene
                else:
                    count = sum([feature_ids.count(i) for i in [21,22,23]])
                    for i in range(count):
                        subtask = ' '.join(['pick', 'up', 'the', 'cup'])
                        subtasks.append(subtask)
            elif tree[subtree_idx, 0] == 'CD':
                count = num_to_int[tokens[get_abs_idx(tree, [subtree_idx,0])].lower()]
                for i in range(count):
                    subtask = ' '.join(['pick', 'up', 'the'] + obj_l[1:-1] + ['cup'])
                    subtasks.append(subtask)
        return subtasks
    
    def generate_place_tasks(tree, subtree_idx):
        subtasks = []
        loc_l = tokens[get_abs_idx(tree,[subtree_idx,0]):get_abs_idx(tree,[subtree_idx,0])+len(tree[subtree_idx])]
        # TODO change once "place" is recognized
        subtask = ' '.join(['pour', 'it'] + loc_l)
        subtasks.append(subtask)
        return subtasks
    
    trees = get_trees(tags)
    if len(trees) > 1:
        print('WARNING: more than one tree generated')
    if len(trees)  == 0:
        print('ERROR: parser could not understand input')
        return []

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
            # TEMP hack to handle second command "put it in the dish" when the robot is holding something
            if tags[1] == 'PRP':
                print('no subtasks generated')
                print([sentence])
                return [sentence]
            
            subtasks += generate_pick_tasks(tree, 1)
            place_subtasks = generate_place_tasks(tree, 2)*len(subtasks)
            # interleave
            subtasks = [val for pair in zip(subtasks, place_subtasks) for val in pair]
            print('generated subtasks:', subtasks)
            return subtasks

        elif len(tree) == 5:
            # V O CC O L
            subtasks = []
            
            subtasks += generate_pick_tasks(tree, 1)
            subtasks += generate_pick_tasks(tree, 3)
            place_subtasks = generate_place_tasks(tree, 4)*len(subtasks)
            # interleave
            subtasks = [val for pair in zip(subtasks, place_subtasks) for val in pair]
            print('generated subtasks:', subtasks)
            return subtasks

        elif len(tree) == 6:
            # V O CC V O L
            subtasks = []
            
            subtasks += generate_pick_tasks(tree, 1)
            place_subtasks = generate_place_tasks(tree, 5)*len(subtasks)
            # interleave
            subtasks = [val for pair in zip(subtasks, place_subtasks) for val in pair]
            print('generated subtasks:', subtasks)
            return subtasks

        elif len(tree) == 7:
            # V O CC O CC O L
            return [] # TODO
        

# def _parser_old(sentence):
#     action = None
#     # pattern matching
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

#     # step 1: segment sentence based on POS tags
#     segments = {'V': [['VB', 'RP'], ['VB']],
#     'L': [['IN', 'DT', 'JJ', 'NN'], ['IN', 'DT', 'NN']],
#     'O': [['DT', 'JJ', 'NN'], ['DT', 'NN'], ['PDT', 'DT', 'JJ', 'NNS'], ['PDT', 'DT', 'NNS'], ['PDT', 'JJ', 'NNS'], ['PDT', 'NNS'], ['CD', 'JJ', 'NN'], ['CD', 'JJ', 'NNS'], ['CD', 'NN'], ['CD', 'NNS'], ['PRP']]}
#     segmented = tags[:]
#     for pattern_name, patterns in segments.items():
#         for pattern in patterns:
#             while pattern_finder(segmented, pattern, pattern_name) != segmented:
#                 segmented = pattern_finder(segmented, pattern, pattern_name)
#     print(segmented)

#     # step 2: identify action
#     templates = {'pick': [['V','O']],
#                  'place': [['V','O','L'],['V','O','CC','O','L'],['V','O','CC','O','CC','O','L']],
#                  'pick_and_place': [['V','O','CC','V','O','L']]}
    
#     for action_name, actions in templates.items():
#         for template in actions:
#             if segmented == template:
#                 action = action_name
#                 print('action:', action_name)
#                 print(segmented)
#                 print(sentence)
    
#     if action is None:
#         print('ERROR: sentence is not correctly structured')
#         return []
    
#     # step 3: generate subtask commands
#     if action == 'pick':
#         print('no subtasks generated: ')
#         print([sentence])
#         return [sentence]
#     else:
#         loc = tokens[tags.index('IN'):] # ['in','the','bin']
#         objs = tokens[tags.index('')]
#         subtasks = []
        



    # outputs:
    # "pick up the red cup" [VB RP DT JJ NN]: V(PICK_UP) O_0
    # "place it in the red bin" [VB PRP IN DT JJ NN]: V(PLACE) O_0/PRP L




if __name__ == "__main__":
    sentences = [
        "pick up the cup",
        "put it in the bin",
        "put the green cup in the red bin",
        "pick up the cup and put it in the bin",
        "put all the yellow cups in the red bin",
        "place two red cups and one green cup in the red bin"
    ]
    for sentence in sentences:
        semantic_parser(sentence)