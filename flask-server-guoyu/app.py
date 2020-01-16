import getopt
import os
import sys
from time import sleep

from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

sourceA_index = '1'
_sourceA_path = 'static/image_fusion/animalsA/1.jpg'
_img = Image.open(_sourceA_path)
_img.save('static/source-image-fusion-1.jpg')
sourceB_index = '1'
_sourceB_path = 'static/image_fusion/animalsB/1.jpg'
_img = Image.open(_sourceB_path)
_img.save('static/source-image-fusion-2.jpg')
_targetAB_path = 'static/image_fusion/animalsC/1-1.jpg'
_img = Image.open(_targetAB_path)
_img.save('static/target-image-fusion-1-2.jpg')


def option_prepare(argv):
    args_dict_ = {}
    opts, args = getopt.getopt(argv, "", ['host=', 'port=', 'debug=', 'options='])
    for opt, arg in opts:
        if opt == "--host":
            args_dict_['host'] = arg
        elif opt == "--port":
            args_dict_['port'] = int(arg)
        elif opt == "--debug":
            args_dict_['debug'] = arg
        elif opt == "--options":
            args_dict_['options'] = arg
    return args_dict_


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route('/')
def main():
    return render_template('style-transfer.html')


@app.route('/style-transfer.html')
def style_transefer():
    return render_template('style-transfer.html')


@app.route('/style-transfer-process', methods=['post'])
def style_transfer_process():
    sleep(1)
    style = request.form['style']
    source_pic = request.form['source'].split('.')[0]
    source_ext = request.form['source'].split('.')[1]
    style_dir = style.split(';')[0]
    style_pic = style.split(';')[1].split('.')[0]
    style_ext = style.split(';')[1].split('.')[1]
    target_pic = source_pic + '-' + style_pic
    target_ext = source_ext
    source_path = 'static/content/' + source_pic + '.' + source_ext
    img = Image.open(source_path)
    img.save('static/source-style-transfer.png')
    target_path = 'static/outputs4/' + target_pic + '-2.0.' + target_ext
    img = Image.open(target_path)
    img.save('static/target-style-transfer.png')
    return render_template('style-transfer.html')


@app.route('/segmentation.html')
def segmentation():
    return render_template('segmentation.html')


@app.route('/segmentation-process', methods=['post'])
def segmentation_process():
    sleep(1)
    source_pic = request.form['source']
    target_pic = request.form['source'].split('_')[0] + '_pred.png'
    source_path = 'static/segmentation/raw/' + source_pic
    img = Image.open(source_path)
    img.save('static/source-segmentation.png')
    target_path = 'static/segmentation/pred/' + target_pic
    img = Image.open(target_path)
    img.save('static/target-segmentation.png')
    return render_template('segmentation.html')


@app.route('/matting.html')
def matting():
    return render_template('matting.html')


@app.route('/matting-process', methods=['post'])
def matting_process():
    sleep(1)
    source_pic = request.form['source']
    target_pic = request.form['source']
    source_path = 'static/matting/images/' + source_pic
    img = Image.open(source_path)
    img.save('static/source-matting.png')
    matting_path = 'static/matting/mattes/' + target_pic
    img = Image.open(matting_path)
    img.save('static/trimap-matting.png')
    target_path = 'static/matting/colors/' + target_pic
    img = Image.open(target_path)
    img.save('static/target-matting.png')
    return render_template('matting.html')


@app.route('/image-fusion.html')
def image_fusion():
    return render_template('image-fusion.html')


@app.route('/image-fusion-process', methods=['post'])
def image_fusion_process():
    sleep(1)
    global sourceA_index, sourceB_index
    print(request.form)
    if request.form.get('image-fusion-source1') is not None and request.form["image-fusion-source1"] != '':
        sourceA_pic = request.form["image-fusion-source1"]
        sourceA_path = 'static/image_fusion/animalsA/' + sourceA_pic
        sourceA_index = sourceA_pic.split('.')[0]
        img = Image.open(sourceA_path)
        img.save('static/source-image-fusion-1.jpg')
    if request.form.get('image-fusion-source2') is not None and request.form["image-fusion-source2"] != '':
        sourceB_pic = request.form["image-fusion-source2"]
        sourceB_path = 'static/image_fusion/animalsB/' + sourceB_pic
        sourceB_index = sourceB_pic.split('.')[0]
        img = Image.open(sourceB_path)
        img.save('static/source-image-fusion-2.jpg')
    if request.form.get('fusion') is not None and request.form.get('fusion') != '':
        target_pic = sourceA_index + '-' + sourceB_index + '.jpg'
        targetAB_path = 'static/image_fusion/animalsC/' + target_pic
        img = Image.open(targetAB_path)
        img.save('static/target-image-fusion-1-2.jpg')
    print('okk')
    return render_template('image-fusion.html')


if __name__ == '__main__':
    args_dict = option_prepare(sys.argv[1:])
    print(args_dict)
    app.run(host=args_dict['host'], port=args_dict['port'])
