import cv2
import numpy as np
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\sohai\\PycharmProjects\\Server\\handwriting-8e91fd0b8d13.json"


class Words:
    path = ""
    total_box_count = 0
    mapping = []
    max_score = 600
    wordCache = ""

    def __init__(self,path):
        self.path = path

    def set_cache(self, word):
        self.wordCache = word

    def process_image2(self,correct,length):

        image = cv2.imread(self.path)
        scale_percent = 50  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        alpha = 0.5
        overlay = image.copy()
        output = image.copy()
        height, width, channels = image.shape
        print(correct)
        #height always 600
        correct = ErrorGradient(correct)
        print(correct)
        adding = width/len(correct)
        start = 0
        print(correct)
        for i in correct:
            cv2.rectangle(overlay, (start, 0), (int(start + adding), height), (0, 255*(i), (255*(1-i)) ), -1)

            start += int(adding)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        #cv2.imshow("output",output)
        #cv2.waitKey(0)
        cv2.imwrite('Updated_change.jpg', output)
        # cv2.imshow("output",output)
        # cv2.waitKey(0)
        return image


    def detect_document(self):
        """Detects document features in an image."""
        final_string = ""
        self.max_score = 600
        self.mapping.clear()
        from google.cloud import vision
        import io
        client = vision.ImageAnnotatorClient()

        with io.open(self.path, 'rb') as image_file:
            content = image_file.read()

        image = vision.types.Image(content=content)

        response = client.document_text_detection(image=image)
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                print('\nBlock confidence: {}\n'.format(block.confidence))
                final_string += ("\nBlock confidence : {} \n".format(block.confidence))

                for paragraph in block.paragraphs:
                    print('Paragraph confidence: {}'.format(
                        paragraph.confidence))
                    final_string += ('Paragraph confidence: {} '.format(
                        paragraph.confidence))

                    for word in paragraph.words:
                        word_text = ''.join([
                            symbol.text for symbol in word.symbols
                        ])
                        print('Word text: {} (confidence: {})'.format(
                            word_text, word.confidence))
                        final_string += ('Word text: {} (confidence: {})'.format(
                            word_text, word.confidence))
                        a = 0.5
                        b = 0.9
                        if(word.confidence > 0 and word.confidence < a):
                            self.max_score = 0.1 * self.max_score
                        elif word.confidence >= a and word.confidence < 1:
                            self.max_score = self.max_score*float(np.exp(word.confidence-a)-b)

                        for symbol in word.symbols:
                            print('\tSymbol: {} (confidence: {})'.format(
                                symbol.text, symbol.confidence))
                            final_string += ('\tSymbol: {} (confidence: {})'.format(
                                symbol.text, symbol.confidence))
                            self.mapping.append(("{}".format(symbol.text), "{}".format(symbol.confidence)))

        print("----------------------------------------")
        print(self.mapping)
        return final_string

    def calculateIncorrectLetters(self,word):
        self.detect_document()
        #self.mapping = [("e",1),("m",1),("x",1),("i",1),("x",1)]
        correct = []

        print(self.total_box_count,len(self.mapping))
        start = 0
        for i in range(0,len(word)):
            try:
                if word[0].lower() == self.mapping[i][0].lower():
                    start = i
                    break
                else:
                    correct.append(0)
            except IndexError:
                correct.append(0)
        counter = 0
        for i in range(start,len(self.mapping)):
            try:
                if word[counter].lower() == self.mapping[i][0].lower():
                    correct.append(1)
                else:
                    correct.append(0)
            except IndexError:
                correct.append(0)
            counter += 1

        return self.process_image2(correct, len(word))


import numpy as np
import matplotlib.pyplot as plt


def ErrorGradient(bin_array):

    word_length = len(bin_array)
    per_letter = 10
    print(len(bin_array))
    new = []# np.zeros((1,per_letter*word_length))

    current = 0
    for current in range(0,len(bin_array)-1):

        if bin_array[current] == bin_array[current + 1]:
            new.append(bin_array[current]*np.ones((per_letter)))


        elif bin_array[current] != bin_array[current+1]:
            new.append(np.linspace(bin_array[current],bin_array[current+1],per_letter))


    new = np.reshape(new,(1,per_letter*(len(bin_array)-1)))

    def smooth(x,window_len=11,window='hanning'):


        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')
        return y

    y = smooth(new[0],window_len = word_length, window = 'hamming')
    return y



if __name__ == '__main__':
    temp = Words("C:\\Users\\sohai\\PycharmProjects\\Server\\words_rec.jpg")
    temp.calculateIncorrectLetters("email")

