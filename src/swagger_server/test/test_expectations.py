#from swagger_server.test.tester import TestPosts  # noqa: E501
import click
from click.testing import CliRunner
import pandas as pd
import great_expectations as ge
import sys
sys.path.insert(0, '.')
from swagger_server.util import get_config_data

@click.command()
@click.option('--config', prompt='config', help='config file')
def api_cli(config):
    models, features = get_config_data(config)
    click.echo(click.style(
        f'{models.__len__()} model(s) are loaded', fg='green'))


def test_cli():
    runner = CliRunner()
    result = runner.invoke(api_cli, ['--config', 'config_v4.json'])
    assert result.exit_code == 0
    assert result.output == '2 model(s) are loaded\n'


def test_data():
    my_df = ge.read_csv("./swagger_server/test/test_cases.csv")  #.fillna(0)
    # https://colab.research.google.com/github/datarootsio/tutorial-great-expectations/blob/main/tutorial_great_expectations.ipynb#scrollTo=F7HjCbmsQlPE
    assert my_df.expect_column_to_exist('cohort_id').success is True
    assert my_df.expect_column_values_to_not_be_null(
        column="trop0").success is True
    assert my_df.expect_column_values_to_be_of_type(
        column="trop0", type_="int64").success is True
    # expected_columns = ["Unnamed: 0", "cohort_id","ds","supercell_id","subjectid","trop0","trop1","trop2","trop3","trop4","trop5","trop6","time_trop0","time_trop1","time_trop2","time_trop3","time_trop4","time_trop5","time_trop6","avgtrop","avgspd","maxtrop","mintrop","maxvel","minvel","divtrop","difftrop","diffvel","logtrop0","phys_albumin","phys_bnp","phys_ckmb","phys_creat","phys_crp","phys_dimer","phys_ferritin","phys_fibrin","phys_haeglob","phys_hba1c","phys_lacta","phys_lactv","phys_pco2","phys_ph","phys_platec","phys_platev","phys_po2","phys_tsh","phys_urate","phys_urea","phys_wbc","priorami","prioracs","priorangina","priorvt","priorcva","priorrenal","priorsmoke","priorcopd","priorpci","priorcabg","priordiab","priorhtn","priorhf","priorarrhythmia","priorhyperlipid","gender","age","angiogram","mdrd_gfr","out5","out3c","outl1","outl2","event_mi","event_t1mi","event_t2mi","event_t4mi","event_t5mi","event_dead","event_dmi30d","quantized_trop_0-2","quantized_trop_2-4","quantized_trop_4-6","quantized_trop_6-8","quantized_trop_8-10","quantized_trop_10-12","quantized_trop_12-14","quantized_trop_14-16","quantized_trop_16-18","quantized_trop_18-20","quantized_trop_20-22","quantized_trop_22-24","time_trop7","trop8","time_trop8","trop7","set"]
    # assert my_df.expect_table_columns_to_math_ordered_list(column_list=expected_columns).success is True
    # assert my_df.expect_table_column_values_to_not_be_null(column="cohort_id").success is True
    # assert my_df.expect_table_column_values_to_be_of_type(column="trop0", type_="float64").success is True
