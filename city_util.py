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


class ResInfo( object ):
    def __init__( self, name, rix, pb_name, base_prod, nr_name, bb_name, lake_name ):
        self.name = name
        self.rix = rix
        self.pb_name = pb_name # prod building name 
        self.base_prod = base_prod
        self.nr_name = nr_name # natural res node name
        self.bb_name = bb_name # boost building name
        self.lake_name = lake_name # 'lake' type booster res name (if applicable)

        
all_res = [ ResInfo( "wood",  0, "woodcutter", 300, "wood", "sawmill", None ),
            ResInfo( "stone", 1, "quarry", 300, "stone", "stonemason", None ),
            ResInfo( "iron",  2, "iron_mine", 300, "iron", "foundry", None ),
            ResInfo( "food",  3, "farm", 300, "building_place_on_ground", "mill", "lake" ),
            ]

nr_names = set( [res.nr_name for res in all_res] 
                + [res.lake_name for res in all_res if res.lake_name] )
pb_names = set( res.pb_name for res in all_res )


class ParseError( ValueError ):
    pass

class CityLayout( object ):
    city_rows = 21
    city_cols = 21
    city_elems = city_rows*city_cols
    # set the templates to None for bootstrapping parse_ncp_str()
    land_template = None
    water_template = None

    def __init__( self ):
        self.kind = None
        self.template = None
        self.output_str_func = self.get_fcp_str
        self._mat = [None]*self.city_elems
        self._res_mat = [None]*self.city_elems*len(all_res) # for res placment scores
    def res_mat_get( self, i, j, rix ):
        return self._res_mat[ i + j*self.city_cols + rix*self.city_elems ]
    def res_mat_set( self, i, j, rix, v ):
        self._res_mat[ i + j*self.city_cols + rix*self.city_elems ] = v
    def res_mat_invalidate( self, s_i, s_j ):
        dist = 2
        for j in xrange(s_j-dist,s_j+dist+1):
            if j>0 and j<self.city_rows:
                for i in xrange(s_i-dist,s_i+dist+1):
                    if i>0 and i<self.city_cols:
                        for rix in xrange(len(all_res)):
                            self.res_mat_set(i,j,rix,None)
        
    def mat_get( self, i, j ):
        return self._mat[ i + j*self.city_cols ]
    def mat_set( self, i, j, v ):
        self._mat[ i + j*self.city_cols ] = v
    def iter_mat( self ):
        ix = 0
        for j in xrange( self.city_rows ):
            for i in xrange( self.city_cols ):
                yield i,j,self._mat[ix]
                ix += 1


    # note: the results includes s_i,s_j itself
    def iter_adj( self, s_i, s_j, dist = 1 ):
        for j in xrange(s_j-dist,s_j+dist+1):
            if j>0 and j<self.city_rows:
                for i in xrange(s_i-dist,s_i+dist+1):
                    if i>0 and i<self.city_cols:
                        yield i,j,self._mat[ i + j*self.city_cols ]#mat_get(i,j)

    def check_len_and_parse_kind( self, err_str, s, expected_len, char_kind_map ):
        if len(s) != expected_len:
            raise ParseError( "bad city %s str length (whitespace removed) %s, expected %s" 
                                % (err_str, len(s),expected_len) )
        self.kind = char_kind_map.get(s[0:1],None)
        if self.kind is None:
            raise ParseError( "bad fcp city type: %s, expected something in %s" %
                                (s[0:1],char_kind_map.keys() ) )
        self.template = self.land_template
        if self.kind == "water_city": self.template = self.water_template

    def parse_ncp_str( self, ncp_str, ex=False ):
        ncp_str = re.sub(r'\s', '', ncp_str)
        ss_strip_res = re.search( r'\[ShareString.*?\]([^[]*)(?:\[/ShareString.*\])?', ncp_str )
        if ss_strip_res: ncp_str = ss_strip_res.group(1)
        expected_len = 1 + self.city_elems
        self.check_len_and_parse_kind( "ncp", ncp_str, expected_len, ncp_char_kind_map )
        pos = 1
        for i,j,cur_cobj in self.iter_mat():
            read_cobj = None
            if ex:
                # note: ex=1 is only called internally, so no nice user error check here.
                read_cobj = ncp_ex_char_cobj_map[ ncp_str[pos] ]
            else:
                try:
                    read_cobj = ncp_char_cobj_map[ ncp_str[pos] ]
                except KeyError, e:
                    raise ParseError( "bad ncp city object char %s, expected something in %s" % 
                                      (ncp_str[pos],ncp_char_cobj_map.keys() ) )
                if read_cobj == False: read_cobj = self.template.mat_get(i,j)
            assert not read_cobj is None
            self.mat_set(i,j,read_cobj)
            pos += 1
        self.output_str_func = self.get_ncp_str
        return self # syntax sugar

    # just try parsing as ncp first, then try as fcp
    def parse_str( self, s ):
        try:
            return self.parse_ncp_str( s )
        except ParseError, ncp_e:
            try:
                return self.parse_fcp_str( s )
            except ParseError, fcp_e:
                raise ParseError( ("failed to parse %r as either fcp or ncp format:\n"
                                   "ncp error: %s\n"
                                   "fcp_error: %s\n" ) % (s, ncp_e,fcp_e) )

    # wrt our matrix, the fcp format only has data for things that are
    # not walls in the templates. we use that fact to calc the correct
    # length, and later in the iteration to fill in the correct
    # elements. note that both the land and water templates have walls
    # in the same places, so we arbitrarily use the land template for
    # the wall checks.
    def parse_fcp_str( self, fcp_str ):
        lt = CityLayout.land_template # used for wall checks
        fcp_str = re.sub(r'\s|\.', '', fcp_str)
        url_strip_res = re.search( r'map=(.*)', fcp_str )
        if url_strip_res: fcp_str = url_strip_res.group(1)
        expected_len = 1 + sum( cobj != "wall" for i,j,cobj in lt.iter_mat() )
        self.check_len_and_parse_kind( "fcp", fcp_str, expected_len, fcp_char_kind_map )
        pos = 1
        for i,j,lt_cobj in lt.iter_mat():
            if lt_cobj != "wall":
                read_cobj = fcp_char_cobj_map.get( fcp_str[pos], None )
                if read_cobj == False: read_cobj = self.template.mat_get(i,j)
                if read_cobj is None:
                    raise ParseError( "bad fcp city object char %s, expected something in %s"
                                        % (fcp_str[pos],fcp_char_cobj_map.keys()) )
                self.mat_set(i,j,read_cobj)
                pos += 1
            else:
                self.mat_set(i,j,lt_cobj)
        self.output_str_func = self.get_fcp_str
        return self # syntax sugar

    def get_ncp_str( self, readable = False ):
        ret = "" if readable else "[ShareString.1.3]"
        nl = "\n" if readable else ""
        ret += ncp_kind_char_map[self.kind] + nl
        for j in xrange(self.city_rows):
            ret += "".join( ncp_cobj_char_map[self.mat_get(i,j)] for i in xrange(self.city_cols) ) + nl
        ret += "" if readable else "[/ShareString]"
        return ret

    def get_fcp_str( self, readable = False ):
        ret = "" if readable else fcp_url_prefix
        nl = "\n" if readable else ""
        ret += fcp_kind_char_map[self.kind] + nl
        for j in xrange(self.city_rows):
            for i in xrange(self.city_cols):
                c = fcp_cobj_char_map[self.mat_get(i,j)]
                if not c and readable: c = '.'
                ret += c
            ret += nl
        return ret

    def get_str( self, readable = False ):
        return self.output_str_func( readable )

    def remove_non_nr( self ):
        for i,j,cobj in self.iter_mat():
            if not cobj in nr_names:
                self.mat_set(i,j,self.template.mat_get(i,j))

    def remove_unlocked_nrs( self, locked ):
        for i,j,cobj in self.iter_mat():
            if (not (i,j) in locked) and (cobj in nr_names) and (cobj != "lake"):
                self.mat_set(i,j,self.template.mat_get(i,j))
                
    def calc_res( self ):
        ret = ""
        ret += "--- res totals ---\n"
        all_tot = 0
        for res in all_res:
            res_tot = 0 if res.name != "wood" else 300 # hack for TH town_hall wood prod
            for i,j,cobj in self.iter_mat():
                res_tot += self.calc_res_at( res, i, j )
            all_tot += res_tot
            ret += "%s %s\n" % (res.name, res_tot )

        tot_buildings = 0
        tot_cottages = 0
        for i,j,cobj in self.template.iter_mat():
            if ( (cobj == "building_place_on_ground") and
                 (self.mat_get(i,j) != cobj) and
                 (not self.mat_get(i,j) in nr_names) ):
                tot_buildings += 1
                if self.mat_get(i,j) == "cottage":
                    tot_cottages += 1
        prod_per_b = "nan"
        if tot_buildings:
            prod_per_b = float(all_tot)/tot_buildings
        ret += ( "--- end res --- cottages %s other_buildings %s total_buildings %s"
                 " all_tot %s prod/building %s ---" %
                (tot_cottages,tot_buildings-tot_cottages,tot_buildings, all_tot, prod_per_b ) )
        return ret

    def calc_res_at( self, res, i, j ):
        if self.mat_get(i,j) == res.pb_name:
            adj = [ cobj for i,j,cobj in self.iter_adj(i,j) ]
            prod = res.base_prod
            num_nr = adj.count( res.nr_name )
            prod = int( prod * ( 1 + (num_nr*.4) + int(num_nr>0)*.1 ) ) # nat res
            prod = int( prod * ( 1 + adj.count("cottage")*.3 ) ) # cottages
            prod = int( prod * ( 1 + adj.count(res.lake_name)*.5 ) ) # lakes
            prod = int( prod * ( 1 + int(res.bb_name in adj)*.75 ) ) # boost building
            return prod
        else:
            return 0

    def can_build_land( self, i, j, locked ):
        if self.build_only_on_open:
            return( (self.mat_get(i,j) == "building_place_on_ground") and
                    not ( (i,j) in locked ) )
        else:
            return( (self.template.mat_get(i,j) == "building_place_on_ground") and
                    not ( (i,j) in locked ) )

    def greedy_place_prod_all_res( self, use_slots = 72, num_cottages = 15, 
                                   keep_extra_res_nodes = 0, build_only_on_open = 0,
                                   placement_schedule = [ "WSI" ] ):
        # sigh, i'm too lazy to pass this down:
        self.build_only_on_open = build_only_on_open

        # hacky optimization: if there are < 3 nat res (originally) in
        # range of a potential booster spot, don't bother. we don't
        # update the cands as res are destroyed.
        self.res_bb_cands = dict()
        for res in all_res:
            res_bb_cands = []
            self.res_bb_cands[res.name] = res_bb_cands
            for i,j,cobj in self.iter_mat():
                if sum( int(cobj==res.nr_name) for i,j,cobj in self.iter_adj(i,j,2) ) >= 3:
                    res_bb_cands.append( (i,j,cobj) )

        assert use_slots > num_cottages
        slots_left = use_slots - num_cottages
        locked = set()
        place_wave = 0
        while slots_left > 0:
            wave_res_set = placement_schedule[ place_wave % len(placement_schedule) ]
            orig_slots_left = slots_left
            slots_left = self.greedy_place_prod_res( locked, slots_left, wave_res_set )
            if orig_slots_left == slots_left:
                break # no progress exit
            place_wave += 1

        slots_left += num_cottages
        for min_cot_adj in xrange(3,-1,-1):
            slots_left = self.greedy_place_cottages( locked, min_cot_adj, slots_left )

        if not keep_extra_res_nodes:
            self.remove_unlocked_nrs( locked )
        #for res in all_res:
        #    if 0 and res.nr_name == "building_place_on_ground":
        #        self.greedy_place_prod_res( res, locked )

    def greedy_place_cottages( self, locked, score_thresh, slots_left ):
        for i,j,cobj in self.iter_mat():
            if self.can_build_land(i,j,locked):
                score = sum( int(cobj in pb_names) for ii,jj,cobj in self.iter_adj(i,j) )
                if score >= score_thresh and slots_left:
                    locked.add( (i,j) )
                    self.mat_set(i,j,"cottage")
                    slots_left -= 1
        return slots_left
                    
    def greedy_place_prod_res( self, locked, slots_left, res_set ):
        all_best = [0,[],None]
        wave_res = [ res for res in all_res if res.name[0].upper() in res_set ]
        for res in wave_res:
            for i,j,cobj in self.res_bb_cands[res.name]:
                if self.can_build_land(i,j,locked) or self.mat_get(i,j) == res.bb_name:
                    best = self.res_mat_get( i, j, res.rix )
                    if (best is None) or (len(best[1]) > slots_left):
                        best = [0,None]
                        already_placed = not self.can_build_land(i,j,locked)
                        placed = []
                        if not already_placed:
                            locked.add( (i,j) )
                            orig_cobj = self.mat_get(i,j)
                            self.mat_set(i,j,res.bb_name)
                            placed = [(i,j,res.bb_name)]
                        adj = [ (ii,jj) for ii,jj,cobj in self.iter_adj( i, j ) 
                                if self.can_build_land(ii,jj,locked) ]
                        self.try_prod( res, i, j, adj, locked, placed, best, slots_left )
                        if best[1] is None: best = None # couldn't place anything
                        self.res_mat_set( i, j, res.rix, best )
                        if not already_placed:
                            self.mat_set(i,j,orig_cobj)
                            locked.remove( (i,j) )
                    if (best is not None) and best[0] > all_best[0]:
                        all_best[0:2] = best
                        all_best[2] = res
        #print "slots_left", slots_left, all_best[0:2]
        
        assert len(all_best[1]) <= slots_left
        slots_left -= len(all_best[1])
        #print "slots_left", slots_left, "placed", best
        for i,j,cobj in all_best[1]:
            assert self.can_build_land( i, j, locked )
            locked.add( (i,j) )
            self.mat_set(i,j,cobj)
            self.res_mat_invalidate(i,j)
        res = all_best[2]
        for i,j,cobj in all_best[1]:
            if cobj == res.pb_name:
                for ii,jj,cobj in self.iter_adj(i,j):
                    if cobj == res.nr_name:
                        locked.add( (ii,jj) )
        return slots_left

    def try_prod( self, res, s_i, s_j, adj, locked, placed, best, slots_left ):
        if len(adj): # recursive case
            i,j = adj[0]
            assert self.can_build_land(i,j,locked) 
            locked.add( (i,j) )
            self.try_prod( res, s_i, s_j, adj[1:], locked, placed, best, slots_left ) # 'don't place' case
            orig_cobj = self.mat_get(i,j)

            if len(placed) < slots_left:
                self.mat_set(i,j,res.pb_name)
                placed.append( (i,j,res.pb_name) )
                self.try_prod( res, s_i, s_j, adj[1:], locked, placed, best, slots_left ) # 'place' case
                placed.pop()

            self.mat_set(i,j,orig_cobj)
            locked.remove( (i,j) )
        elif placed: # leaf case
            tot_res = 0
            for i,j,cobj in placed:
                tot_res += self.calc_res_at( res, i, j )
            score = float(tot_res) / len(placed)
            if score > best[0]:
                best[0] = score
                best[1] = list(placed)
            
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

fcp_res_ex = "http://www.lou-fcp.co.uk/map.php?map=L0000C000000CC00AA0000A00000C0BAAA00BB00000C0000000000B00C0D0000000C0000B0D00D0000C00B0BB00000000C00000BB00A0C00CAAA00B0000A00CC0C00000000DAAA0C000000000D00C00000000A000000000A0CCCC0A000AAA00A0000C000000AA0000000000000000A000D00C0000AA0B000BB000000000A000000000000B0A000000000000C0000CC000000CC"

fcp_res_ex_placed = "http://www.lou-fcp.co.uk/map.php?map=L0000C000000CC00AA0000A00044C0BAAAO3BB00000CM0000222L3B00C0D0000000CKOO3B0D00D04O4C00BOBB000004M4C00333BB0KA0C44CAAA0LB3302220CC0C222033L0DAAA0CKO00OMO0OKO0C00044440A000222000A0CCCC0A000AAA00A0444C0000222A0000MO00000OKOO2A000D00C02K2AA0B000BB0000002O2A2K0000000000B0A0O2000000000C0000CC000000CC"

# these tests are pretty limited due to lack of good input examples
# that exercise all the differs cobjs and such. however, they're a
# start. 
class TestCityLayout(unittest.TestCase):
    def setUp(self):
        pass
    fcp_strs = [ (fcp_water_ex,True), (fcp_ex_1,False), (fcp_res_ex,False),
                 (fcp_res_ex_placed,False) ]
    ncp_strs = [ (ncp_water_ex,True), (ncp_ex_1,False) ]
    all_strs = fcp_strs + ncp_strs

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

    def test_calc_res(self):
        for s,readable in [(fcp_res_ex,False)]:#self.all_strs:
            cl = CityLayout().parse_str( s )
            print cl.calc_res()
            cl.remove_non_nr()
            print cl.calc_res()
            print cl.get_fcp_str()
            cl.greedy_place_prod_all_res( placement_schedule=["W","S","I"])
            print cl.calc_res()
            print cl.get_str()
            
    def test_bad_parse(self):
        self.assertRaises( ParseError, CityLayout().parse_str, "foo" )

if __name__ == '__main__':
    pass
    unittest.main()

