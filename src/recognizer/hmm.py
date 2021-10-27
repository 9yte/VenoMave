import numpy as np
import torch
import multiprocessing
import recognizer.tools as tools

TASKS = ['TIDIGITS', 'SPEECHCOMMANDS']
TASKS_WORDS = {
    'TIDIGITS': {
        'name': ['sil', 'oh', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
        "size": [1, 6, 10, 10, 6, 9, 9, 9, 10, 10, 6, 9],
        'gram': [100000, 1, 100000, 100000, 100000, 100000, 100000, 100000, 10000, 100000, 100000, 100000],
    },
    'SPEECHCOMMANDS': {
        'name': ['sil', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'backward',
                 'bed', 'bird', 'cat', 'dog', 'down', 'follow', 'forward', 'go', 'happy', 'house', 'learn', 'left',
                 'marvin', 'no', 'off', 'on', 'right', 'sheila', 'stop', 'tree', 'up', 'visual', 'wow', 'yes'],
        "size": [1, 10, 10, 6, 9, 9, 9, 10, 10, 6, 9, 18, 9, 9, 9, 9, 9, 12, 18, 6, 12, 9, 9, 12, 18, 6, 6, 6, 9, 12,
                 12, 9, 6, 21, 6, 9],
        'gram': [100000, 1, 100000, 100000, 100000, 100000, 100000, 100000, 10000, 100000, 100000, 100000, 100000,
                 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000,
                 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000, 100000],
    }
}


def _viterbi_train(stateSequence, A):
    '''get the viterbi-posteriors for the statesequence,
    accumulate transition counts in A (extended transition matrix [S+1][S+1])'''

    posteriors = np.zeros((len(stateSequence), len(A)))
    t = 0  # frame index
    prev = 0  # initial state
    A_increase = np.zeros_like(A)
    for state in stateSequence:
        posteriors[t, state] = 1  # mark posterior on best path
        A_increase[prev, state] += 1  # increase transition count
        prev = state
        t += 1

    A_increase[state, 0] += 1
    return A_increase, posteriors


def parallel_decoding(data):
    posteriors, true_length, text, hmm = data
    posteriors = posteriors[:true_length]

    stateSequence, pstar = hmm.viterbi_decode(posteriors)
    word_seq = hmm.getTranscription(stateSequence)

    ref_seq = text.split()

    # apply viterbi training only if transcription is correct
    if np.array_equal(ref_seq, word_seq):
        A_count_increase, y = _viterbi_train(stateSequence, hmm.A_count)
        # y = torch.from_numpy(y).long().cuda()

        # sometimes the targets dimensions differ (by one frame)
        if true_length != y.shape[0]:
            diff = y.shape[0] - true_length
            y = y[:true_length, :]

            if diff > 1:
                raise ValueError('Frame difference larger than 1!')

        return A_count_increase, y

    else:
        return None, None


class HMM:
    iBm = []
    ini = []
    ent = []
    esc = []
    A_count = []

    def __init__(self, task, mode='word'):

        self.mode = mode
        assert task in TASKS
        if self.mode == 'word':
            self.words = TASKS_WORDS[task]
        # elif self.mode == 'phoneme':
        #     self.words = {
        #         'name': ['sp', 'sil', 'OW1', 'Z', 'IH1', 'R', 'OW0', 'IY1', 'HH', 'W', 'AH1', 'N', 'T',
        #                  'UW1', 'TH', 'F', 'AO1', 'AY1', 'V', 'S', 'K', 'EH1', 'AH0', 'EY1'],
        #         'size': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #         'gram': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #     }
        # else:
        #     assert False
        self.words['gram'] = [x / sum(self.words['gram']) for x in self.words['gram']]

        N = sum(self.words['size'])
        self.iBm = [x for x in range(N)]

        a = np.cumsum(self.words['size'])
        self.ini = list(zip([0] + a[:-1].tolist(), self.words['gram']))
        self.ent = list(zip(a[0:], self.words['name'][1:]))
        self.esc = list(zip(a[1:] - 1, self.words['name'][1:]))

        # init Pi vector with zeros, first sil state = 0
        self.piVec = [0] * len(self.iBm)
        self.piVec[self.words['name'].index('sil')] = 1

        self.A = np.zeros((N, N))
        # basic L/R transition structure
        for n in range(0, N):
            self.A[n, n] = 0.5
            self.A[n - 1, n] = 0.5

        self.A = self.modifyTransitions(self.A)
        self.A_count = np.ceil(self.A)

    def get_num_states(self):
        return sum(self.words['size'])

    def modifyTransitions(self, A):

        rowSum = A.sum(axis=1, keepdims=True)
        np.seterr(divide='ignore', invalid='ignore')
        A[:] = np.where(rowSum > 0, A / rowSum, 0.);

        a = np.cumsum(self.words['size']).tolist()
        a00 = A[0][0]  # keep sil
        for row in [x - 1 for x in a]:  # esc
            aii = A[row][row]  # self-transition
            for col, scr in zip([0] + a[:-1], self.words['gram']):  # ini
                A[row][col] = (1.0 - aii) * scr  # remaining prob

        A[0][0] = a00 * (1. + self.words['gram'][0])  # repair sil

        return A

    def input_to_state(self, input):
        ''' maps phone to corresponding HMM state

        Input:

        input:  phone


        Output:

        itervals:   HMM state of input

        '''
        if input not in self.words['name']:
            raise Exception('Undefined word/phone: {}'.format(input))

        # start index of each word
        start_idx = np.insert(np.cumsum(self.words['size']), 0, 0)

        idx = self.words['name'].index(input)

        start_state = start_idx[idx]
        len_phone = self.words['size'][idx]
        end_state = start_state + len_phone

        return [n for n in range(start_state, end_state)]

    def limLog(self, x):
        MINLOG = 1e-100
        return np.log(np.maximum(x, MINLOG))

    def forced_align(self, posteriors, transcription, SPEECH_WEIGHT=None):

        # get target state sequence
        states = []
        [states.append(item) for item in self.input_to_state('sil')]
        for c in transcription:
            [states.append(item) for item in self.input_to_state(c)]
            [states.append(item) for item in self.input_to_state('sil')]

        logA = self.limLog(self.A)
        logPi = self.limLog(self.piVec)

        logpost = self.limLog(posteriors)
        logLike = logpost[:, self.iBm]

        N = len(states)
        T = len(logLike)
        phi = -float('inf') * np.ones((T, N))
        psi = np.zeros((T, N)).astype(int)

        if SPEECH_WEIGHT is None:
            SPEECH_WEIGHT = 0.5

        # Initialize
        for i, state in enumerate(states):
            phi[0, i] = logLike[0, state] + logPi[state]

        # Iteration
        for t in range(1, T):
            for idx, j in enumerate(states):

                if j == 0:
                    w = self.limLog(1 - SPEECH_WEIGHT)
                else:
                    w = self.limLog(SPEECH_WEIGHT)

                mx = -float('inf')

                if idx > 0 and states[idx - 1] == 0:
                    start = max(0, idx - 2)
                else:
                    start = max(0, idx - 1)

                for i in range(start, idx + 1):
                    if phi[t - 1, i] + w + logA[states[i], j] > mx:
                        mx = phi[t - 1, i] + w + logA[states[i], j]
                        phi[t, idx] = mx
                        psi[t, idx] = i

            for i, state in enumerate(states):
                phi[t, i] += logLike[t, state]

        # sort most likely paths (the first one does not necesarily contain all target words)
        idx = np.argsort(phi[t])

        transcription_upper = [word.upper() for word in transcription]
        for i in reversed(idx):
            t = T - 1
            pstar = phi[t, i]
            iopt = psi[t, i]

            # Backtracking
            best_path_t = np.zeros(T).astype(int)
            best_path_t[t] = iopt
            for t in reversed(range(T - 1)):
                iopt = psi[t + 1, iopt]
                best_path_t[t] = iopt

            # map target states to hmm states
            best_path = [states[elem] for elem in best_path_t]

            # verify if targer transcription is decoded and return
            if transcription_upper == self.getTranscription(best_path):
                return best_path, pstar

        return -1, 0

    def viterbi_decode(self, posteriors):
        logPi = self.limLog(self.piVec)
        logA = self.limLog(self.A)

        logpost = self.limLog(posteriors)
        logLike = logpost[:, self.iBm]

        N = len(logPi)
        T = len(logLike)
        phi = -float('inf') * np.ones((T, N))
        psi = np.zeros((T, N)).astype(int)
        best_path = np.zeros(T).astype(int)

        # Initialisierung
        phi[0, :] = logPi + logLike[0]

        # Iteration
        for t in range(1, T):
            for j in range(N):
                mx = -float('inf')
                for i in range(N):
                    if phi[t - 1, i] + logA[i, j] > mx:
                        mx = phi[t - 1, i] + logA[i, j]
                        phi[t, j] = mx
                        psi[t, j] = i

            phi[t, :] += logLike[t]

            # find best result in last column
        t = T - 1
        iopt = 0
        pstar = -float('inf')
        for n in range(N):
            if phi[t, n] > pstar:
                pstar = phi[t, n]
                iopt = n

        # Backtracking
        best_path[t] = iopt
        while t > 1:
            iopt = psi[t, iopt]
            best_path[t - 1] = iopt
            t = t - 1
        # result
        return best_path, pstar

    # get the word sequence for a state sequence and supress state(0) and non-exit states'''
    # exit states is a dictionary with exitState index as key and model name as value
    def getTranscription(self, statesequence):
        '''get the word sequence for a state sequence and supress state(0) and non-exit states'''
        word_list = [''] * len(self.iBm)

        ent = []
        esc = []
        for e_ent, e_esc in zip(self.ent, self.esc):
            word_list[e_ent[0]] = [e_ent[1].upper(), 'entry']
            word_list[e_ent[0] + 1] = [e_ent[1].upper(), 'entry']

            word_list[e_esc[0]] = [e_esc[1].upper(), 'exit']
            word_list[e_esc[0] - 1] = [e_esc[1].upper(), 'exit']

            # entry states
            esc.append(e_esc[0])
            # exit states
            ent.append(e_ent[0])

        # remove duplicate states
        short_state_sequence = []
        prev = -1
        for state in statesequence:
            if state != prev:  # remove duplicate states
                short_state_sequence.append(state)
                prev = state

        words = []
        prev = -1
        for i, state in enumerate(short_state_sequence):
            if word_list[state] != '':  # if state is a valid exit state
                words.append(word_list[state])

        # break here, if no transcription has been found
        if len(words) == 0:
            return -1

        word_sequ = []
        previous_word = words[0]
        for word in words[1:]:
            if previous_word[1] == 'entry' and word[1] == 'exit' and previous_word[0] == word[0]:
                word_sequ.append(previous_word[0])
            previous_word = word

        return word_sequ

    # def _viterbi_train(self, stateSequence, A):
    #     '''get the viterbi-posteriors for the statesequence,
    #     accumulate transition counts in A (extended transition matrix [S+1][S+1])'''
    #
    #     posteriors = np.zeros((len(stateSequence), len(A)))
    #     t = 0  # frame index
    #     prev = 0  # initial state
    #     for state in stateSequence:
    #         posteriors[t, state] = 1  # mark posterior on best path
    #         A[prev, state] += 1  # increase transition count
    #         prev = state
    #         t += 1
    #
    #     A[state, 0] += 1  # exit prob
    #     return A, posteriors
    #
    # def viterbi_train_feat(self, x, y, target_dir, model):
    #
    #     x_tmp = x.data.numpy()
    #
    #     tmp = model.features_to_posteriors(x)
    #
    #     stateSequence, _ = self.viterbi_decode(tmp)
    #
    #     word_seq = self.getTranscription(stateSequence)
    #
    #     lab_dir = str(target_dir).replace('TextGrid', 'lab')
    #     ref_seq = open(lab_dir).read().strip().split(' ')
    #
    #     # apply viterbi training only if transcription is correct
    #     if np.array_equal(ref_seq, word_seq):
    #         self.A_count, y = self._viterbi_train(stateSequence, self.A_count)
    #
    #     return x, y

    def viterbi_train(self, posteriors, y_true_length, ref_labels, text, n_jobs=multiprocessing.cpu_count()):

        with multiprocessing.Pool(n_jobs) as p:
            res = p.map(parallel_decoding, zip(posteriors, y_true_length, text, [self] * len(text)))

        for idx, (A_count_increase, y) in enumerate(res):
            if A_count_increase is not None and y is not None:
                y = torch.from_numpy(y).long().cuda()

                # we don't want to change the padded part (which is silence). We only care about the real decoded part!
                ref_labels[idx, :len(y)] = y

                self.A_count += A_count_increase

        return ref_labels

    def posteriors_to_words(self, posteriors):
        best_path, pstar = self.viterbi_decode(posteriors)

        return self.getTranscription(best_path), best_path
