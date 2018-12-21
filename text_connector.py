from text_line_builder import TextProposalConnector
import numpy as np


def text_connect(MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM, MIN_RATIO, LINE_MIN_SCORE, text_proposals, scores, im_size):
    text_proposal_connector = TextProposalConnector(
        MAX_HORIZONTAL_GAP, MIN_V_OVERLAPS, MIN_SIZE_SIM)

    text_recs = text_proposal_connector.get_text_lines(
        text_proposals, scores, im_size)
    
    keep_inds=filter_boxes(text_recs, MIN_RATIO, LINE_MIN_SCORE)

    return text_recs[keep_inds]


def filter_boxes(boxes, MIN_RATIO, LINE_MIN_SCORE):
    heights = np.zeros((len(boxes), 1), np.float)
    widths = np.zeros((len(boxes), 1), np.float)
    scores = np.zeros((len(boxes), 1), np.float)
    index = 0
    for box in boxes:
        heights[index] = (abs(box[5]-box[1])+abs(box[7]-box[3]))/2.0+1
        widths[index] = (abs(box[2]-box[0])+abs(box[6]-box[4]))/2.0+1
        scores[index] = box[8]
        index += 1
    return np.where((widths/heights > MIN_RATIO) & (scores > LINE_MIN_SCORE))[0]
