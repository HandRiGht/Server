import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\\Users\\sohai\\PycharmProjects\\Server\\handwriting-8e91fd0b8d13.json"


def detect_document(path):
    """Detects document features in an image."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.document_text_detection(image=image)

    if len(response.full_text_annotation.pages) is 0:
        print("null*************************")
        response = client.text_detection(image=image)
        texts = response.text_annotations

        print('Texts:')
        print(texts)
        for text in texts:
            print('\n"{}"'.format(text.description))

            vertices = (['({},{})'.format(vertex.x, vertex.y)
                         for vertex in text.bounding_poly.vertices])

            print('bounds: {}'.format(','.join(vertices)))

    else:
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                print('\nBlock confidence: {}\n'.format(block.confidence))

                for paragraph in block.paragraphs:
                    print('Paragraph confidence: {}'.format(
                        paragraph.confidence))

                    for word in paragraph.words:
                        word_text = ''.join([
                            symbol.text for symbol in word.symbols
                        ])
                        print('Word text: {} (confidence: {})'.format(
                            word_text, word.confidence))

                        for symbol in word.symbols:
                            print('\tSymbol: {} (confidence: {})'.format(
                                symbol.text, symbol.confidence))



if __name__ == '__main__':
    detect_document('C:\\Users\\sohai\\PycharmProjects\\Server\\elephant2.jpeg')


