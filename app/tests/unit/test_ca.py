#%%
from app.src.d01_ana.analysis import Config, Analysis
from app.models import Doc
import pytest

@pytest.fixture(scope='module')
def ca_extensions(test_app):
    settings_analysis = {
    'debug': False,
    'sample': None,
    'clf_model': 'joeddav/xlm-roberta-large-xnli',
    # 'corpus': ['plenar'],
    'pipeline': ['extensions']
    }

    content_analysis = Analysis('test', Config(**settings_analysis))
    content_analysis(to_disk=False, to_db=True)

    res = content_analysis.get_results()
    # res.prepare()
    return res



@pytest.fixture(scope='module')
def content_analysis(test_app):
    settings_analysis = {
    'debug': False,
    'sample': None,
    'clf_model': 'joeddav/xlm-roberta-large-xnli',
    # 'corpus': ['plenar'],
    'pipeline': ['extensions', 'sentiment', 'entity', 'res']
    #     'pipeline': ['extensions', 'sentiment', 'entity', 'res', 'spans', 'clf']
    }

    content_analysis = Analysis('test', Config(**settings_analysis))
    content_analysis(to_disk=False, to_db=True)
    res = content_analysis.get_results()

    # dummy clf coding
    for doc in res.viz:
        for hit in doc:
            hit.E = True
            hit.V = True

    # prepare results
    res.prepare()
    return res

    # load docs with results from db (to_db=True?)
    # tdoc_base = Doc.query.filter_by(id=1).first()
    # tdoc_afd = Doc.query.filter_by(id=3).first()
    # res.coding_pop()
    # res.create_df()
    # res.compute_score_spans()

    # run tests

def test_extensions(ca_extensions):
    assert ca_extensions.doclens == [30, 7, 708, 4]
    # assert ca_extensions.viz == []

def test_labels(content_analysis):
    assert content_analysis.labels == [1, 2, 3, 4]

def test_doclens(content_analysis):
    assert content_analysis.doclens == [30, 7, 708, 4]

def test_lemma(content_analysis):
    assert content_analysis.viz[0][4].lemma == 'betrügen'

def test_span_is_pop(content_analysis):
    assert content_analysis.viz[0][0].SPAN_IS_POP == True

def test_number_of_pop_spans(content_analysis):
    assert len(content_analysis.spans_dict[3]) == 5

def test_load_res_from_db(content_analysis):
    tdoc_base = Doc.query.filter_by(id=1).first()
    tdoc_afd = Doc.query.filter_by(id=3).first()
    assert len(tdoc_afd.res) == 35
    assert len(tdoc_base.res) == 7
    assert content_analysis.viz[0][4].lemma == 'betrügen'
    assert tdoc_base.res[4].lemma == 'betrügen'
    assert tdoc_base.res[4].SPAN_IS_POP == True

def test_load_res_from_fixture(testdoc_afd):
    assert len(testdoc_afd.res) == 35

def test_compare_text_afd(testdoc_afd):
    assert testdoc_afd.text == 'Sehr verehrte Frau Präsident . Sehr verehrte Damen . Sehr geehrte Herren . Liebe Zuschauer . Natürlich gibt es kein explizit sogenanntes Neutralitätsgebot, aber es gibt das journalistische Ethos . die Forderung nach Ausgewogenheit, nach parteipolitischer Neutralität, die Trennung von Nachricht und Meinung . Das gilt für alle Medien und ganz besonders auch für den öffentlich rechtlichen Rundfunk . Mittels der Medien findet öffentliche Meinungsbildung statt . Sie sollen die Informationsfreiheit gewährleisten . Aber leisten sie das heutzutage noch . Heute agieren die Medien und leider auch der öffentlich rechtliche Rundfunk häufig übergriffig und indoktrinierend, auch gegenüber Kindern Kindern, die aufgrund ihrer Unmündigkeit unter einem besonderen Schutz stehen . Wenn aber eine Nachrichtensendung für Kinder ausgesprochen parteipolitisch agiert, dann kann man das nur noch Propaganda nennen, zum Beispiel, wenn in einer KiKA Sendung des ZDF gesagt wird, dass es eine Partei gebe, die Menschen mit anderer Hautfarbe und anderen Religionen hassen würde . Mit einer solchen Sendung wird ein freiheitliches, demokratisches Tabu gebrochen . Hier wird eine Grenze überschritten, die nicht überschritten werden durfte . Schreien Sie nicht so . Schon lange erkennt man folgendes Stereotyp . Je länger Ihre aktuelle, falsch angelegte Politik betrieben wird, umso größer wird der berechtigte Zorn des Bürgers . aber umso ärger der berechtigte Zorn des Bürgers wird, umso größer die Hemmungslosigkeit, mit der Sie, die Sie hier sitzen, unsere Gesellschaft spalten, unsere Nation zerstören und unserer Demokratie unzumutbare Schäden zufügen, und das leider im Zusammenwirken mit den Medien . Ihre aktuelle Politik stellt nur noch eine irrwitzige, irrationale und für den gesunden Menschenverstand nicht mehr nachvollziehbare Ideologie dar, Ihre international sozialistische, kulturmarxistische, etatistische, die im Duktus des hypermoralisierenden Weltenretters daherkommt, weniger weil diese multikulturellen Erlösungsprediger das Fremde so sehr lieben, mehr weil sie das Eigene so sehr hassen . Genau das ist Ihre Botschaft, die Sie hier mit Ihren medialen Helfershelfern ebenso konstant wie hysterisch in allen politischen Feldern rausposaunen ununterbrochen . Vielfaltswahn, Toleranz bis hin zur Selbstaufgabe, Gender Gaga, Klimahysterie, Dieselirrsinn . alles zusammen ein Irrsinn, gegenüber dem der Bürger sich nicht mehr wehren darf, dessen Kosten er aber tragen soll . Wenn aber eine permanent propagierte Ideologie die Redaktionsstuben erobert, dann wird es heikel . Wenn offensichtlich linksideologisch motivierte Journalisten zunehmend für sogenannten Haltungsjournalismus plädieren und diesen praktizieren, dann wird es gefährlich . Umfragen bestätigen, dass annähernd 75 Prozent aller deutschen Journalisten sich selbst dem politisch sehr linken und links grünen Spektrum zuordnen . Der vom Verfassungsgericht geforderte Binnenpluralismus der öffentlich rechtlichen Medien wird so unterlaufen und ausgehebelt . Staatsfern soll er sein und somit frei jeder staatlichen Korrekturmöglichkeit . Das Instrument die Korrekturmöglichkeit, die Freiheit von der Korrektur ist richtig, aber es ist nicht mehr geeicht, meine Damen und Herren . Und die Rundfunkräte, die ja auch aus den parlamentarischen Reihen kommen, haben kein Interesse, das Kontrollinstrument der Aufsicht neu zu eichen . Das schadet der Demokratie, das schadet der Meinungs und Informationsfreiheit . Das fördert die gewünschte Meinung und bestraft die kritische, falsche Meinung . Die gar nicht mehr so unabhängigen Haltungsjournalisten dienen nicht mehr dem Bürger und kontrollieren nicht mehr die Regierenden, sondern umgekehrt . Die Haltungsjournalisten kontrollieren den Bürger und dienen den Regierenden . Meinungsfreiheit war gestern . Die Politiker und ihre Mediengenossen haben ein Klima der Meinungsangst in unserem Land geschaffen . Wir hatten letzten Sonntag, am Abend, in Leipzig den Fall, dass Linksterroristen eine junge Frau zu Hause überfallen haben und diese brutal ins Gesicht geschlagen haben, nur weil sie für ein kapitalistisches Bauunternehmen gearbeitet hat oder arbeitet . Ich habe mein ganzes Berufsleben auch im Medienbereich verbracht . Deshalb ende ich mit einem Zitat von Joseph Pulitzer . Sie kennen ihn alle . Er hat den Pulitzerpreis für Journalisten ausgelobt . Er sagte einst . Eine zynische, käufliche, demagogische Presse wird mit der Zeit ein Volk erzeugen, das genauso niederträchtig ist wie sie selbst . Das angesprochene Vorkommnis in Leipzig bestätigt genau dieses Zitat . Danke schön .\n'

def test_expect_wrong_top_spans(content_analysis):
    with pytest.raises(AssertionError) as excinfo:
        assert content_analysis.top_spans()[0][1] == (0, 100)

# COMPARE EACH PIPELINE STEP! create pickle files in notebook for each element!

# def test_top_span(content_analysis):
#     assert content_analysis.top_spans()[0][1] == (1246, 1688)

# def test_top_spans(content_analysis):
#     assert content_analysis.top_spans() == [
#         (3, (1246, 1688), 51.327767683142405),
#         (1, (0, 159), 42.16132100528045),
#         (3, (1410, 2093), 16.088684300958167),
#         (3, (4236, 4585), 13.836609275886591),
#         (3, (2356, 2892), 12.743573484856206),
#         (3, (1689, 2473), 12.665681079388783)]


