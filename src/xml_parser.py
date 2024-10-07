import xml.etree.ElementTree as ET

class XMLParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.tree = ET.parse(self.file_path)
        self.root = self.tree.getroot()

    def extract_sentences(self):
        """
        Extracting offers and tokens from an XML file
        """
        sentences = self.root.findall(".//sentence")
        X = []
        Y = []

        for sent in sentences:
            X.append([])
            Y.append([])
            for token in sent.iter('token'):
                X[-1].append(token.attrib['text'])

            tags_obj_list = sent.findall(".//token/tfr/v/l/g[1]")
            for tag in tags_obj_list:
                Y[-1].append(tag.get('v'))

        return X, Y

    def convert_to_text(self, token_list):
        """
        Converting tokens to a string
        """
        text = ""
        for tokens in token_list:
            text += " ".join(tokens) + " "
        return text