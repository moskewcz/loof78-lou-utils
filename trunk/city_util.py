ncp_kind_char_map = { "land_city":":", "water_city":";" }
ncp_char_kind_map = dict((v,k) for k, v in ncp_kind_char_map.iteritems())
assert( len( ncp_kind_char_map ) == len( ncp_char_kind_map ) )
# it would be nice if ncp distinguished walls from water, as we need
# this info to convert to/from fcp format. for our internal templates,
# we customize the format a little to handle this -- see the fcp
# parser for details.
ncp_cobj_char_map = {
    "stone":":",
    "wood":".",
    "iron":",",
    "lake":";",
    "woodcutter":"2",
    "quarry":"3",
    "farm":"1",
    "cottage":"C",
    "market":"P",
    "iron_mine":"4",
    "sawmill":"L",
    "mill":"M",
    "hideout":"H",
    "stonemason":"A",
    "foundry":"D",
    "town_hall":"T",
    "townhouse":"U",
    "barracks":"B",
    "city_guard_house":"K",
    "training_ground":"G",
    "stable":"E",
    "workshop":"Y",
    "shipyard":"V",
    "warehouse":"S",
    "castle":"X",
    "harbor":"R",
    "moonglow_tower":"J",
    "trinsic_temple":"Z",
    "blocked":"#",
    "building_place_on_ground":"-",
    "building_place_on_water":"_",
    "old_woodcutter":"W",
    "old_quarry":"Q",
    "old_iron_mine":"I",
    "old_farm":"F"
}
ncp_char_cobj_map = dict((v,k) for k, v in ncp_cobj_char_map.iteritems())
assert( len( ncp_char_cobj_map ) == len( ncp_cobj_char_map ) )
assert "#" not in ncp_cobj_char_map
ncp_char_cobj_map["#"] = False # indicates we should use template for this spot
ncp_cobj_char_map.update( {
        "wall":"#",
        "water_tile_in_water_city":"#",
        } )

fcp_kind_char_map = { "land_city":"L", "water_city":"W" }
fcp_char_kind_map = dict((v,k) for k, v in fcp_kind_char_map.iteritems())
fcp_cobj_char_map = {
    "stone":"B",
    "wood":"A",
    "iron":"C",
    "lake":"D",
    "woodcutter":"2",
    "quarry":"3",
    "farm":"5",
    "cottage":"O",
    "market":"J",
    "iron_mine":"4",
    "sawmill":"K",
    "mill":"N",
    "hideout":"1",
    "stonemason":"L",
    "foundry":"M",
    "townhouse":"E",
    "barracks":"P",
    "city_guard_house":"S",
    "training_ground":"Q",
    "stable":"U",
    "workshop":"V",
    "shipyard":"Y",
    "warehouse":"Z",
    "castle":"X",
    "harbor":"T",
    "moonglow_tower":"R",
    "trinsic_temple":"W",
    "old_woodcutter":"F",
    "old_quarry":"G",
    "old_iron_mine":"H",
    "old_farm":"I"
}
fcp_char_cobj_map = dict((v,k) for k, v in fcp_cobj_char_map.iteritems())
assert( len( fcp_char_cobj_map ) == len( fcp_cobj_char_map ) )
assert "0" not in fcp_cobj_char_map
fcp_char_cobj_map["0"] = False # indicates we should use template for this spot
fcp_cobj_char_map.update( {
        "town_hall":"0",
        "wall":"", 
        "water_tile_in_water_city":"0",
        "building_place_on_ground":"0",
        "building_place_on_water":"0",
        } )
fcp_url_prefix = "http://www.lou-fcp.co.uk/map.php?map="

# for our 'extened' version of the ncp format we use to store our
# internal templates, we only need this limited mapping
ncp_ex_char_cobj_map = {
    "-":"building_place_on_ground",
    "_":"building_place_on_ground",
    "T":"town_hall","9":"water_tile_in_water_city","#":"wall"}

ncp_ex_water_template = """;
#####################
###-------#-------###
##--------#--------##
#---------#---------#
#---------#---------#
#------#######------#
#-----##-----##-----#
#----##-------##----#
#----#---------#----#
#----#---------#----#
######----T----######
#----#---------#----#
#----#---------#----#
#----##-------##----#
#-----##-----##-----#
#------#######--__--#
#---------#----_99_-#
#---------#----_999_#
##--------#-----_99##
###-------#------_###
#####################
"""

ncp_ex_land_template = """:
#####################
###-------#-------###
##--------#--------##
#---------#---------#
#---------#---------#
#------#######------#
#-----##-----##-----#
#----##-------##----#
#----#---------#----#
#----#---------#----#
######----T----######
#----#---------#----#
#----#---------#----#
#----##-------##----#
#-----##-----##-----#
#------#######------#
#---------#---------#
#---------#---------#
##--------#--------##
###-------#-------###
#####################
"""

import re, unittest

class CityLayout( object ):
    city_rows = 21
    city_cols = 21
    # set the templates to None for bootstrapping parse_ncp_str()
    land_template = None
    water_template = None
    @classmethod
    def get_template_for_kind( self, kind ):
        template = self.land_template
        if kind == "water_city": template = self.water_template
        return template

    def __init__( self ):
        self.kind = None
        self.mat = [ [None]*self.city_cols for i in range(0,self.city_rows) ]

    def check_len_and_parse_kind( self, err_str, s, expected_len, char_kind_map ):
        if len(s) != expected_len:
            raise RuntimeError( "bad city %s str length (whitespace removed) %s, expected %s" 
                                % (err_str, len(s),expected_len) )
        self.kind = char_kind_map.get(s[0:1],None)
        if self.kind is None:
            raise RuntimeError( "bad fcp city type: %s, expected something in %s" %
                                (s[0:1],char_kind_map.keys() ) )
        return CityLayout.get_template_for_kind( self.kind )

    def parse_ncp_str( self, ncp_str, ex=False ):
        ncp_str = re.sub(r'\s', '', ncp_str)
        ss_strip_res = re.search( r'\[ShareString.*?\]([^[]*)(?:\[/ShareString.*\])?', ncp_str )
        if ss_strip_res: ncp_str = ss_strip_res.group(1)
        expected_len = 1 + sum( len(row) for row in self.mat )
        template = self.check_len_and_parse_kind( "ncp", ncp_str, expected_len, ncp_char_kind_map )
        pos = 1
        for j,row in enumerate(self.mat):
            for i in xrange(0,len(row)):
                if ex:
                    # note: ex=1 is only called internally, so no nice user error check here.
                    row[i] = ncp_ex_char_cobj_map[ ncp_str[i+pos] ]
                else:
                    row[i] = ncp_char_cobj_map.get( ncp_str[i+pos], None )
                    if row[i] == False: row[i] = template.mat[j][i]
                if row[i] is None:
                    raise RuntimeError( "bad ncp city object char %s, expected something in %s"
                                        % (ncp_str[i+pos],ncp_char_cobj_map.keys()) )
            pos += len(row)
        return self # syntax sugar

    # wrt our matrix, the fcp format only has data for things that are
    # not walls in the templates. we use that fact to calc the correct
    # length, and later in the iteration to fill in the correct
    # elements. note that both the land and water templates have walls
    # in the same places, so we arbitrarily use the land template for
    # the wall checks.
    def parse_fcp_str( self, fcp_str ):
        lt_mat = CityLayout.land_template.mat # used for wall checks
        fcp_str = re.sub(r'\s|\.', '', fcp_str)
        url_strip_res = re.search( r'map=(.*)', fcp_str )
        if url_strip_res: fcp_str = url_strip_res.group(1)
        expected_len = 1 + sum( sum( cobj != "wall" for cobj in row ) for row in lt_mat )
        template = self.check_len_and_parse_kind( "fcp", fcp_str, expected_len, fcp_char_kind_map )
        pos = 1
        for j,row in enumerate(lt_mat):
            for i in xrange(0,len(row)):
                if row[i] != "wall":
                    cobj = fcp_char_cobj_map.get( fcp_str[pos], None )
                    if cobj == False: cobj = template.mat[j][i]
                    if cobj is None:
                        raise RuntimeError( "bad fcp city object char %s, expected something in %s"
                                            % (fcp_str[pos],fcp_char_cobj_map.keys()) )
                    self.mat[j][i] = cobj
                    pos += 1
                else:
                    self.mat[j][i] = row[i]
        return self # syntax sugar

    def get_ncp_str( self, readable = False ):
        ret = "" if readable else "[ShareString.1.3]"
        nl = "\n" if readable else ""
        ret += ncp_kind_char_map[self.kind] + nl
        for row in self.mat:
            ret += "".join( ncp_cobj_char_map[cobj] for cobj in row ) + nl
        ret += "" if readable else "[/ShareString]"
        return ret

    def get_fcp_str( self, readable = False ):
        ret = "" if readable else fcp_url_prefix
        nl = "\n" if readable else ""
        ret += fcp_kind_char_map[self.kind] + nl
        for j,row in enumerate(self.mat):
            for i in xrange(0,len(row)):
                c = fcp_cobj_char_map[row[i]]
                if not c and readable: c = '.'
                ret += c
            ret += nl
        return ret
    
# for fcp str handling, we need templates. we generate them from the ncp templates above.
CityLayout.land_template = CityLayout().parse_ncp_str( ncp_ex_land_template, ex=True )
CityLayout.water_template = CityLayout().parse_ncp_str( ncp_ex_water_template, ex=True )


## begin uint test data

fcp_water_ex = """W
.....................
...0000000.000ZZZZ...
..0000BB00.000LKLK0..
.000000B00.0D0ZZZZZZ.
.CCC0B0BB0.000B0NMMM.
.00C0B0.......0ZZZZZ.
.0000B..0000D..0NNNN.
.AA00..ZKZMZLZ..ZZZZ.
.0A00.0ZKZMZLZ0.NNNN.
.0AA0.0ZKZMZLZ0.ZZZZ.
......0ZKZ0ZLZ0......
.0000.AZKZMZLZ0.R000.
.0CC0.AZKZMZLZB.0000.
.0000..ZKZMZLZ..0000.
.00000..00C00..00JJJ.
.A00000.......000T00.
.AAA00BBB0.00D0000TJ.
.0000BB000.D000T0000.
..00000BB0.0000JT00..
...0000000.0B0JJJ0...
.....................
"""

fcp_ex_1 = "http://www.lou-fcp.co.uk/map.php?map=W0000ZZ0000AA0B000000000C0000000D0000ZZ00000000000000000000000000CC0OOO00AA00005IID0PPPPPA0000DNOO0PRPRP0000A0II5PPPPPPPPP0AAA0OODP0PRPRPRPA000PPPP0PPPP0000P0PRPRPRPPPPD0000PPPPPPPPPP00000000PRPRP0P00C00000PPPPPPPPP0000000PP00PP000D000C0AA0P0000P000000000000P000000000000AAAAPP00000000000000PP0"

ncp_water_ex = """;
#####################
###-------#---SSSS###
##----::--#---ALAL-##
#------:--#-;-SSSSSS#
#,,,-:-::-#---:-MDDD#
#--,-:-#######-SSSSS#
#----:##----;##-MMMM#
#..--##SLSDSAS##SSSS#
#-.--#-SLSDSAS-#MMMM#
#-..-#-SLSDSAS-#SSSS#
######-SLSTSAS-######
#----#.SLSDSAS-#J---#
#-,,-#.SLSDSAS:#----#
#----##SLSDSAS##----#
#-----##--,--##--PPP#
#.-----#######--_R--#
#...--:::-#--;-_##RP#
#----::---#;---R###_#
##-----::-#----PR####
###-------#-:-PPP_###
#####################
"""
ncp_ex_1 = "[ShareString.1.3];########################----SS-#---..-:#####--------#-,------###-;----SS-#---------##---------#-------,,##-CCC--#######..----##1FF;-##BBBBB##.----##;MCC##-BJBJB-##---.##-FF1#BBBBBBBBB#-...##-CC;#B-BJBJBJB#.---#######BBBBTBBBB#######----#B-BJBJBJB#BBB;##----#BBBBBBBBB#B---##----##-BJBJB-##B--,##-----##BBBBB##BBBB-##------#######BB__BB##---;---,-#..-B_##_B##---------#---B_###_###-------.#...BB_#######-------#----BB_########################[/ShareString]"

# these tests are pretty limited due to lack of good input examples
# that exercise all the differs cobjs and such. however, they're a
# start. 
class TestCityLayout(unittest.TestCase):
    def setUp(self):
        pass
    fcp_strs = [ (fcp_water_ex,True), (fcp_ex_1,False) ]
    ncp_strs = [ (ncp_water_ex,True), (ncp_ex_1,False) ]

    def test_fcp_io(self):
        for fcp_str,readable in self.fcp_strs:
            cl = CityLayout().parse_fcp_str( fcp_str )
            self.assertEqual( fcp_str, cl.get_fcp_str( readable ) )
    def test_ncp_io(self):
        for ncp_str,readable in self.ncp_strs:
            cl = CityLayout().parse_ncp_str( ncp_str )
            self.assertEqual( ncp_str, cl.get_ncp_str( readable ) )
    def test_roundtrip_fcp(self):
        for fcp_str,readable in self.fcp_strs:
            cl = CityLayout().parse_fcp_str( fcp_str )
            cl2 = CityLayout().parse_ncp_str( cl.get_ncp_str() )
            self.assertEqual( fcp_str, cl.get_fcp_str( readable ) )
    def test_roundtrip_ncp(self):
        for ncp_str,readable in self.ncp_strs:
            cl = CityLayout().parse_ncp_str( ncp_str )
            cl2 = CityLayout().parse_fcp_str( cl.get_fcp_str() )
            self.assertEqual( ncp_str, cl.get_ncp_str( readable ) )

if __name__ == '__main__':
    unittest.main()

