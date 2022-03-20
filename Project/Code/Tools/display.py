from IPython.display import display, HTML, clear_output

def display_board(board, use_svg=True):
    if use_svg:
        return board._repr_svg_()
    else:
        return "<pre>" + str(board) + "</pre>"
