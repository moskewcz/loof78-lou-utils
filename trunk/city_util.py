
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


#note:
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

import re

class CityLayout( object ):

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
        self.mat = [ [None]*21 for i in range(0,21) ]

    def parse_ncp_str( self, ncp_str, ex=False ):
        template = CityLayout.get_template_for_kind( self.kind )
        ncp_str = re.sub(r'\s', '', ncp_str)
        expected_len = 1 + sum( len(row) for row in self.mat )
        if len(ncp_str) != expected_len:
            raise RuntimeError( "bad ncp_str length (after removing whitespace) %s, expected %s" 
                                % (len(ncp_str),expected_len) )
        self.kind = ncp_char_kind_map.get(ncp_str[0:1],None)
        if self.kind is None:
            raise RuntimeError( "bad ncp city type: %s, expected something in %s" %
                                (ncp_str[0:1],ncp_char_kind_map.keys() ) )
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

    def get_ncp_str( self, readable = False ):
        nl = "\n" if readable else ""
        ret = ncp_kind_char_map[self.kind] + nl
        for row in self.mat:
            ret += "".join( ncp_cobj_char_map[cobj] for cobj in row ) + nl
        return ret

    def get_fcp_str( self, readable = False ):
        nl = "\n" if readable else ""
        ret = fcp_kind_char_map[self.kind] + nl
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


cl = CityLayout()
cl.parse_ncp_str( ncp_water_ex )
print cl.get_ncp_str( readable=True )
print cl.get_fcp_str( readable=True )

