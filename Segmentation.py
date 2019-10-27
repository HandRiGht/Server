import cv2
import numpy as np
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\sohai\\PycharmProjects\\Server\\handwriting-8e91fd0b8d13.json"


class Words:
    path = ""
    total_box_count = 0
    mapping = []
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
        if not all(correct):
            correct = Smoothing(len(correct)* 5,correct)
        else:
            correct = [not x for x in correct]
        adding = width/len(correct)
        start = 0
        print(correct)
        for i in correct:
            cv2.rectangle(overlay, (start, 0), (int(start + adding), height), (0, 255*(1-i), (255*i) ), -1)

            start += int(adding)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

        #cv2.imshow("output",output)
        #cv2.waitKey(0)
        cv2.imwrite('Updated_change.jpg', output)
        cv2.imshow("output",output)
        cv2.waitKey(0)
        return image

    def process_image(self):

        image = cv2.imread(self.path)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        #binary
        ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

        #dilation
        kernel = np.ones((5,5), np.uint8)
        img_dilation = cv2.dilate(thresh, kernel, iterations=1)

        #find contours
        ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

        self.total_box_count = len(sorted_ctrs)
        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv2.boundingRect(ctr)

            # Getting ROI
            roi = image[y:y+h, x:x+w]

            # show ROI
            cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)

        return image


    def detect_document(self):
        """Detects document features in an image."""
        final_string = ""
        self.mapping = []
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
        #self.mapping = [("e",1),("m",1),("a",1),("i",1),("l",1)]
        self.process_image()
        correct = []

        print(self.total_box_count,len(self.mapping))
        for i in range(0,len(word)):
            print(word[i] + " " + self.mapping[i][0])
            if word[i] == self.mapping[i][0]:
                correct.append(1)
            else:
                correct.append(0)

        return self.process_image2(correct, len(word))


import numpy as np
import matplotlib.pyplot as plt


def Smoothing(word_length,i):
    def gaussian(x, mu, sig):
        return (1 / np.sqrt(2 * np.pi) * sig) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    mean = 10
    std = 10
    x = np.linspace(-2 * std, 2 * std, word_length)
    y = gaussian(x, mean, std)


    smoothed = np.convolve(i, y)
##
    smoothed = smoothed  /max(smoothed)
    return list(smoothed)




if __name__ == '__main__':
    temp = Words("C:\\Users\\sohai\\PycharmProjects\\Server\\words_rec.jpg")
    temp.calculateIncorrectLetters("email")

