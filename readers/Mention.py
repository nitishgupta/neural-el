start_word = "<s>"
end_word = "<eos>"
unk_word = "<unk_word>"


class Mention(object):
    def __init__(self, mention_line):
        ''' mention_line : Is the string line stored for each mention
            mid wid wikititle start_token end_token surface tokenized_sentence
            all_types
        '''
        mention_line = mention_line.strip()
        split = mention_line.split("\t")
        (self.mid, self.wid, self.wikititle) = split[0:3]
        self.start_token = int(split[3]) + 1  # Adding <s> in the start
        self.end_token = int(split[4]) + 1
        self.surface = split[5]
        self.sent_tokens = [start_word]
        self.sent_tokens.extend(split[6].split(" "))
        self.sent_tokens.append(end_word)
        self.types = split[7].split(" ")
        if len(split) > 8:    # If no mention surface words in coherence
            if split[8].strip() == "":
                self.coherence = [unk_word]
            else:
                self.coherence = split[8].split(" ")
        if len(split) == 10:
            self.docid = split[9]

        assert self.end_token <= (len(self.sent_tokens) - 1), "Line : %s" % mention_line
    #enddef

    def toString(self):
        """
        Returns a string representation of the sentence.

        Args:
            self: (todo): write your description
        """
        outstr = self.wid + "\t"
        outstr += self.wikititle + "\t"
        for i in range(1, len(self.sent_tokens)):
            outstr += self.sent_tokens[i] + " "

        outstr = outstr.strip()
        return outstr

#endclass
