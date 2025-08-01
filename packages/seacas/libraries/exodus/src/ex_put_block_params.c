/*
 * Copyright(C) 1999-2024 National Technology & Engineering Solutions
 * of Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
 * NTESS, the U.S. Government retains certain rights in this software.
 *
 * See packages/seacas/LICENSE for details
 */

#include "exodusII.h"     // for ex_block, ex_err, etc
#include "exodusII_int.h" // for EX_FATAL, etc
#include <stdbool.h>

/*!
 * writes the parameters used to describe an element/face/edge block
 * \param   exoid                   exodus file id
 * \param   block_count             number of blocks being defined
 * \param   blocks                  array of ex_block structures describing
 * block counts
 */

int ex_put_block_params(int exoid, size_t block_count, const struct ex_block *blocks)
{
  size_t i;
  int    conn_int_type;
  int    status;
  int    arbitrary_polyhedra = 0; /* 1 if block is arbitrary 2d polyhedra type; 2 if 3d polyhedra */
  int    att_name_varid      = -1;
  int    varid, dimid, dims[2], blk_id_ndx, blk_stat, strdim;
  size_t start[2];
  size_t num_blk;
  int    cur_num_blk, numblkdim, numattrdim;
  int    nnodperentdim = -1;
  int    nedgperentdim = -1;
  int    nfacperentdim = -1;
  int    connid        = 0;
  int    npeid;
  char   errmsg[MAX_ERR_LENGTH];
  char  *entity_type1     = NULL;
  char  *entity_type2     = NULL;
  int   *blocks_to_define = NULL;
  const char *dnumblk     = NULL;
  const char *vblkids     = NULL;
  const char *vblksta     = NULL;
  const char *vnodcon     = NULL;
  const char *vnpecnt     = NULL;
  const char *vedgcon     = NULL;
  const char *vfaccon     = NULL;
  const char *vconn       = NULL;
  const char *vattnam     = NULL;
  const char *vblkatt     = NULL;
  const char *dneblk      = NULL;
  const char *dnape       = NULL;
  const char *dnnpe       = NULL;
  const char *dnepe       = NULL;
  const char *dnfpe       = NULL;

  if (block_count == 0) {
    return EX_NOERR;
  }

  EX_FUNC_ENTER();
  if (exi_check_valid_file_id(exoid, __func__) == EX_FATAL) {
    EX_FUNC_LEAVE(EX_FATAL);
  }

  /*
   * ========================================================================
   * Check whether `blocks` is homogeneous (all same type) and if so, does it
   * contain entries for all blocks of that type that will be defined. If so,
   * can consolidate some operations...
   */
  bool all_same  = true;
  int  last_type = blocks[0].type;
  for (i = 0; i < block_count; i++) {
    if (blocks[i].type != last_type) {
      all_same = false;
      break;
    }
    /* See if storing an 'nsided' element block (arbitrary 2d polyhedra or super
     * element) */
    if (strlen(blocks[i].topology) >= 3) {
      if ((blocks[i].topology[0] == 'n' || blocks[i].topology[0] == 'N') &&
          (blocks[i].topology[1] == 's' || blocks[i].topology[1] == 'S') &&
          (blocks[i].topology[2] == 'i' || blocks[i].topology[2] == 'I')) {
        all_same = false;
        break;
      }
      else if ((blocks[i].topology[0] == 'n' || blocks[i].topology[0] == 'N') &&
               (blocks[i].topology[1] == 'f' || blocks[i].topology[1] == 'F') &&
               (blocks[i].topology[2] == 'a' || blocks[i].topology[2] == 'A')) {
        /* If a FACE_BLOCK, then we are dealing with the faces of the nfaced
         * blocks[i]. */
        all_same = false;
        break;
      }
    }
  }

  if (all_same) {
    /*
     * Check number of blocks of this type on the database and
     * see if that is the size of `blocks` array (i.e., are all
     * blocks of that type being defined in this call.
     */
    switch (last_type) {
    case EX_EDGE_BLOCK: dnumblk = DIM_NUM_ED_BLK; break;
    case EX_FACE_BLOCK: dnumblk = DIM_NUM_FA_BLK; break;
    case EX_ELEM_BLOCK: dnumblk = DIM_NUM_EL_BLK; break;
    default:
      snprintf(errmsg, MAX_ERR_LENGTH,
               "ERROR: Bad block type (%d) specified for all blocks file id %d", last_type, exoid);
      ex_err_fn(exoid, __func__, errmsg, EX_BADPARAM);
      EX_FUNC_LEAVE(EX_FATAL);
    }
    if ((status = exi_get_dimension(exoid, dnumblk, ex_name_of_object(last_type), &num_blk, &dimid,
                                    __func__)) != EX_NOERR) {
      snprintf(errmsg, MAX_ERR_LENGTH, "ERROR: No %ss defined in file id %d",
               ex_name_of_object(last_type), exoid);
      ex_err_fn(exoid, __func__, errmsg, status);
      EX_FUNC_LEAVE(EX_FATAL);
    }

    if (block_count == num_blk) {
      status = exi_put_homogenous_block_params(exoid, block_count, blocks);
      EX_FUNC_LEAVE(status);
    }
  }

  if (!(blocks_to_define = malloc(block_count * sizeof(int)))) {
    snprintf(errmsg, MAX_ERR_LENGTH,
             "ERROR: failed to allocate memory for internal blocks_to_define "
             "array in file id %d",
             exoid);
    ex_err_fn(exoid, __func__, errmsg, EX_MEMFAIL);
    EX_FUNC_LEAVE(EX_FATAL);
  }

  for (i = 0; i < block_count; i++) {
    switch (blocks[i].type) {
    case EX_EDGE_BLOCK:
      dnumblk = DIM_NUM_ED_BLK;
      vblkids = VAR_ID_ED_BLK;
      vblksta = VAR_STAT_ED_BLK;
      break;
    case EX_FACE_BLOCK:
      dnumblk = DIM_NUM_FA_BLK;
      vblkids = VAR_ID_FA_BLK;
      vblksta = VAR_STAT_FA_BLK;
      break;
    case EX_ELEM_BLOCK:
      dnumblk = DIM_NUM_EL_BLK;
      vblkids = VAR_ID_EL_BLK;
      vblksta = VAR_STAT_EL_BLK;
      break;
    default:
      snprintf(errmsg, MAX_ERR_LENGTH,
               "ERROR: Bad block type (%d) specified for entry %d file id %d", blocks[i].type,
               (int)i, exoid);
      ex_err_fn(exoid, __func__, errmsg, EX_BADPARAM);
      free(blocks_to_define);
      EX_FUNC_LEAVE(EX_FATAL);
    }

    /* first check if any blocks of that type are specified */
    if ((status = exi_get_dimension(exoid, dnumblk, ex_name_of_object(blocks[i].type), &num_blk,
                                    &dimid, __func__)) != EX_NOERR) {
      snprintf(errmsg, MAX_ERR_LENGTH, "ERROR: No %ss defined in file id %d",
               ex_name_of_object(blocks[i].type), exoid);
      ex_err_fn(exoid, __func__, errmsg, status);
      free(blocks_to_define);
      EX_FUNC_LEAVE(EX_FATAL);
    }

    /* Next: Make sure that there are not any duplicate block ids by
       searching the vblkids array.
       WARNING: This must be done outside of define mode because id_lkup
       accesses
       the database to determine the position
    */

    if ((status = nc_inq_varid(exoid, vblkids, &varid)) != EX_NOERR) {
      snprintf(errmsg, MAX_ERR_LENGTH, "ERROR: failed to locate %s ids in file id %d",
               ex_name_of_object(blocks[i].type), exoid);
      ex_err_fn(exoid, __func__, errmsg, status);
      free(blocks_to_define);
      EX_FUNC_LEAVE(EX_FATAL);
    }

    status = exi_id_lkup(exoid, blocks[i].type, blocks[i].id);
    if (-status != EX_LOOKUPFAIL) { /* found the element block id */
      snprintf(errmsg, MAX_ERR_LENGTH, "ERROR: %s id %" PRId64 " already exists in file id %d",
               ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
      ex_err_fn(exoid, __func__, errmsg, EX_DUPLICATEID);
      free(blocks_to_define);
      EX_FUNC_LEAVE(EX_FATAL);
    }

    /* Keep track of the total number of element blocks defined using a counter
       stored in a linked list keyed by exoid.
       NOTE: exi_get_file_item  is a function that finds the number of element
       blocks for a specific file and returns that value.
    */
    cur_num_blk = exi_get_file_item(exoid, exi_get_counter_list(blocks[i].type));
    if (cur_num_blk >= (int)num_blk) {
      snprintf(errmsg, MAX_ERR_LENGTH, "ERROR: exceeded number of %ss (%d) defined in file id %d",
               ex_name_of_object(blocks[i].type), (int)num_blk, exoid);
      ex_err_fn(exoid, __func__, errmsg, EX_BADPARAM);
      free(blocks_to_define);
      EX_FUNC_LEAVE(EX_FATAL);
    }

    /*   NOTE: exi_inc_file_item  is a function that finds the number of element
         blocks for a specific file and returns that value incremented. */
    cur_num_blk = exi_inc_file_item(exoid, exi_get_counter_list(blocks[i].type));
    start[0]    = cur_num_blk;

    /* write out block id to previously defined id array variable*/
    status = nc_put_var1_longlong(exoid, varid, start, (long long *)&blocks[i].id);

    if (status != EX_NOERR) {
      snprintf(errmsg, MAX_ERR_LENGTH, "ERROR: failed to store %s id to file id %d",
               ex_name_of_object(blocks[i].type), exoid);
      ex_err_fn(exoid, __func__, errmsg, status);
      free(blocks_to_define);
      EX_FUNC_LEAVE(EX_FATAL);
    }

    blocks_to_define[i] = start[0] + 1; /* element id index into vblkids array*/

    if (blocks[i].num_entry == 0) { /* Is this a NULL element block? */
      blk_stat = 0;                 /* change element block status to NULL */
    }
    else {
      blk_stat = 1; /* change element block status to TRUE */
    }

    if ((status = nc_inq_varid(exoid, vblksta, &varid)) != EX_NOERR) {
      snprintf(errmsg, MAX_ERR_LENGTH, "ERROR: failed to locate %s status in file id %d",
               ex_name_of_object(blocks[i].type), exoid);
      ex_err_fn(exoid, __func__, errmsg, status);
      free(blocks_to_define);
      EX_FUNC_LEAVE(EX_FATAL);
    }

    if ((status = nc_put_var1_int(exoid, varid, start, &blk_stat)) != EX_NOERR) {
      snprintf(errmsg, MAX_ERR_LENGTH,
               "ERROR: failed to store %s id %" PRId64 " status to file id %d",
               ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
      ex_err_fn(exoid, __func__, errmsg, status);
      free(blocks_to_define);
      EX_FUNC_LEAVE(EX_FATAL);
    }
  }

  /* put netcdf file into define mode  */
  if ((status = exi_redef(exoid, __func__)) != EX_NOERR) {
    snprintf(errmsg, MAX_ERR_LENGTH, "ERROR: failed to place file id %d into define mode", exoid);
    ex_err_fn(exoid, __func__, errmsg, status);
    free(blocks_to_define);
    EX_FUNC_LEAVE(EX_FATAL);
  }

  for (i = 0; i < block_count; i++) {
    if (blocks[i].num_entry == 0) { /* Is this a NULL element block? */
      continue;
    }

    blk_id_ndx = blocks_to_define[i];

    switch (blocks[i].type) {
    case EX_EDGE_BLOCK:
      dneblk  = DIM_NUM_ED_IN_EBLK(blk_id_ndx);
      dnnpe   = DIM_NUM_NOD_PER_ED(blk_id_ndx);
      dnepe   = NULL;
      dnfpe   = NULL;
      dnape   = DIM_NUM_ATT_IN_EBLK(blk_id_ndx);
      vblkatt = VAR_EATTRIB(blk_id_ndx);
      vattnam = VAR_NAME_EATTRIB(blk_id_ndx);
      vnodcon = VAR_EBCONN(blk_id_ndx);
      vedgcon = NULL;
      vfaccon = NULL;
      break;
    case EX_FACE_BLOCK:
      dneblk  = DIM_NUM_FA_IN_FBLK(blk_id_ndx);
      dnnpe   = DIM_NUM_NOD_PER_FA(blk_id_ndx);
      dnepe   = NULL;
      dnfpe   = NULL;
      dnape   = DIM_NUM_ATT_IN_FBLK(blk_id_ndx);
      vblkatt = VAR_FATTRIB(blk_id_ndx);
      vattnam = VAR_NAME_FATTRIB(blk_id_ndx);
      vnodcon = VAR_FBCONN(blk_id_ndx);
      vnpecnt = VAR_FBEPEC(blk_id_ndx);
      vedgcon = NULL;
      vfaccon = NULL;
      break;
    case EX_ELEM_BLOCK:
      dneblk  = DIM_NUM_EL_IN_BLK(blk_id_ndx);
      dnnpe   = DIM_NUM_NOD_PER_EL(blk_id_ndx);
      dnepe   = DIM_NUM_EDG_PER_EL(blk_id_ndx);
      dnfpe   = DIM_NUM_FAC_PER_EL(blk_id_ndx);
      dnape   = DIM_NUM_ATT_IN_BLK(blk_id_ndx);
      vblkatt = VAR_ATTRIB(blk_id_ndx);
      vattnam = VAR_NAME_ATTRIB(blk_id_ndx);
      vnodcon = VAR_CONN(blk_id_ndx);
      vnpecnt = VAR_EBEPEC(blk_id_ndx);
      vedgcon = VAR_ECONN(blk_id_ndx);
      vfaccon = VAR_FCONN(blk_id_ndx);
      break;
    default: goto error_ret;
    }

    /* define some dimensions and variables*/
    if ((status = nc_def_dim(exoid, dneblk, blocks[i].num_entry, &numblkdim)) != EX_NOERR) {
      if (status == NC_ENAMEINUSE) { /* duplicate entry */
        snprintf(errmsg, MAX_ERR_LENGTH, "ERROR: %s %" PRId64 " already defined in file id %d",
                 ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
      }
      else {
        snprintf(errmsg, MAX_ERR_LENGTH,
                 "ERROR: failed to define number of entities/block for %s %" PRId64 " file id %d",
                 ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
      }
      goto error_ret; /* exit define mode and return */
    }

    if (dnnpe && blocks[i].num_nodes_per_entry > 0) {
      /* A nfaced block would not have any nodes defined... */
      if ((status = nc_def_dim(exoid, dnnpe, blocks[i].num_nodes_per_entry, &nnodperentdim)) !=
          EX_NOERR) {
        snprintf(errmsg, MAX_ERR_LENGTH,
                 "ERROR: failed to define number of nodes/entity for %s %" PRId64 " in file id %d",
                 ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
        goto error_ret; /* exit define mode and return */
      }
    }

    if (dnepe && blocks[i].num_edges_per_entry > 0) {
      if ((status = nc_def_dim(exoid, dnepe, blocks[i].num_edges_per_entry, &nedgperentdim)) !=
          EX_NOERR) {
        snprintf(errmsg, MAX_ERR_LENGTH,
                 "ERROR: failed to define number of edges/entity for %s %" PRId64 " in file id %d",
                 ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
        goto error_ret; /* exit define mode and return */
      }
    }

    if (dnfpe && blocks[i].num_faces_per_entry > 0) {
      if ((status = nc_def_dim(exoid, dnfpe, blocks[i].num_faces_per_entry, &nfacperentdim)) !=
          EX_NOERR) {
        snprintf(errmsg, MAX_ERR_LENGTH,
                 "ERROR: failed to define number of faces/entity for %s %" PRId64 " in file id %d",
                 ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
        goto error_ret; /* exit define mode and return */
      }
    }

    /* element attribute array */
    if (blocks[i].num_attribute > 0) {

      if ((status = nc_def_dim(exoid, dnape, blocks[i].num_attribute, &numattrdim)) != EX_NOERR) {
        snprintf(errmsg, MAX_ERR_LENGTH,
                 "ERROR: failed to define number of attributes in %s %" PRId64 " in file id %d",
                 ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
        goto error_ret; /* exit define mode and return */
      }

      dims[0] = numblkdim;
      dims[1] = numattrdim;

      if ((status = nc_def_var(exoid, vblkatt, nc_flt_code(exoid), 2, dims, &varid)) != EX_NOERR) {
        snprintf(errmsg, MAX_ERR_LENGTH,
                 "ERROR:  failed to define attributes for %s %" PRId64 " in file id %d",
                 ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
        goto error_ret; /* exit define mode and return */
      }
      exi_compress_variable(exoid, varid, 2);

      /* inquire previously defined dimensions  */
      if ((status = nc_inq_dimid(exoid, DIM_STR_NAME, &strdim)) != EX_NOERR) {
        snprintf(errmsg, MAX_ERR_LENGTH, "ERROR: failed to get string length in file id %d", exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
        goto error_ret;
      }

      /* Attribute names... */
      dims[0] = numattrdim;
      dims[1] = strdim;

      if ((status = nc_def_var(exoid, vattnam, NC_CHAR, 2, dims, &att_name_varid)) != EX_NOERR) {
        snprintf(errmsg, MAX_ERR_LENGTH,
                 "ERROR: failed to define %s attribute name array in file id %d",
                 ex_name_of_object(blocks[i].type), exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
        goto error_ret; /* exit define mode and return */
      }
#if defined(EX_CAN_USE_NC_DEF_VAR_FILL)
      int fill = NC_FILL_CHAR;
      nc_def_var_fill(exoid, att_name_varid, 0, &fill);
#endif
    }

    conn_int_type = NC_INT;
    if (ex_int64_status(exoid) & EX_BULK_INT64_DB) {
      conn_int_type = NC_INT64;
    }

    /* See if storing an 'nsided' element block (arbitrary 2d polyhedra or super
     * element) */
    if (strlen(blocks[i].topology) >= 3) {
      if ((blocks[i].topology[0] == 'n' || blocks[i].topology[0] == 'N') &&
          (blocks[i].topology[1] == 's' || blocks[i].topology[1] == 'S') &&
          (blocks[i].topology[2] == 'i' || blocks[i].topology[2] == 'I')) {
        arbitrary_polyhedra = 1;
      }
      else if ((blocks[i].topology[0] == 'n' || blocks[i].topology[0] == 'N') &&
               (blocks[i].topology[1] == 'f' || blocks[i].topology[1] == 'F') &&
               (blocks[i].topology[2] == 'a' || blocks[i].topology[2] == 'A')) {
        /* If a FACE_BLOCK, then we are dealing with the faces of the nfaced
         * blocks[i]. */
        arbitrary_polyhedra = blocks[i].type == EX_FACE_BLOCK ? 1 : 2;
      }
    }

    /* element connectivity array */
    if (arbitrary_polyhedra > 0) {
      if (blocks[i].type != EX_FACE_BLOCK && blocks[i].type != EX_ELEM_BLOCK) {
        snprintf(errmsg, MAX_ERR_LENGTH,
                 "ERROR: Bad block type (%d) for nsided/nfaced block in file id %d", blocks[i].type,
                 exoid);
        ex_err_fn(exoid, __func__, errmsg, EX_BADPARAM);
        goto error_ret;
      }

      if (arbitrary_polyhedra == 1) {
        dims[0] = nnodperentdim;
        vconn   = vnodcon;

        /* store entity types as attribute of npeid variable -- node/elem,
         * node/face, face/elem*/
        entity_type1 = "NODE";
        if (blocks[i].type == EX_ELEM_BLOCK) {
          entity_type2 = "ELEM";
        }
        else {
          entity_type2 = "FACE";
        }
      }
      else if (arbitrary_polyhedra == 2) {
        dims[0] = nfacperentdim;
        vconn   = vfaccon;

        /* store entity types as attribute of npeid variable -- node/elem,
         * node/face, face/elem*/
        entity_type1 = "FACE";
        entity_type2 = "ELEM";
      }

      if ((status = nc_def_var(exoid, vconn, conn_int_type, 1, dims, &connid)) != EX_NOERR) {
        snprintf(errmsg, MAX_ERR_LENGTH,
                 "ERROR: failed to create connectivity array for %s %" PRId64 " in file id %d",
                 ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
        goto error_ret; /* exit define mode and return */
      }

      /* element face-per-element or node-per-element count array */
      dims[0] = numblkdim;

      if ((status = nc_def_var(exoid, vnpecnt, conn_int_type, 1, dims, &npeid)) != EX_NOERR) {
        snprintf(errmsg, MAX_ERR_LENGTH,
                 "ERROR: failed to create face- or node- per-entity "
                 "count array for %s %" PRId64 " in file id %d",
                 ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
        goto error_ret; /* exit define mode and return */
      }

      if ((status = nc_put_att_text(exoid, npeid, "entity_type1", strlen(entity_type1) + 1,
                                    entity_type1)) != EX_NOERR) {
        snprintf(errmsg, MAX_ERR_LENGTH,
                 "ERROR: failed to store entity type attribute text for %s %" PRId64
                 " in file id %d",
                 ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
        goto error_ret; /* exit define mode and return */
      }
      if ((status = nc_put_att_text(exoid, npeid, "entity_type2", strlen(entity_type2) + 1,
                                    entity_type2)) != EX_NOERR) {
        snprintf(errmsg, MAX_ERR_LENGTH,
                 "ERROR: failed to store entity type attribute text for %s %" PRId64
                 " in file id %d",
                 ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
        goto error_ret; /* exit define mode and return */
      }
    }
    else {
      if (blocks[i].num_nodes_per_entry > 0) {
        /* "Normal" (non-polyhedra) element block type */
        dims[0] = numblkdim;
        dims[1] = nnodperentdim;

        if ((status = nc_def_var(exoid, vnodcon, conn_int_type, 2, dims, &connid)) != EX_NOERR) {
          snprintf(errmsg, MAX_ERR_LENGTH,
                   "ERROR: failed to create connectivity array for %s %" PRId64 " in file id %d",
                   ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
          ex_err_fn(exoid, __func__, errmsg, status);
          goto error_ret; /* exit define mode and return */
        }
        exi_compress_variable(exoid, connid, 1);
      }
    }
    /* store element type as attribute of connectivity variable */
    if (connid > 0) {
      if ((status = nc_put_att_text(exoid, connid, ATT_NAME_ELB, strlen(blocks[i].topology) + 1,
                                    blocks[i].topology)) != EX_NOERR) {
        snprintf(errmsg, MAX_ERR_LENGTH, "ERROR: failed to store %s type name %s in file id %d",
                 ex_name_of_object(blocks[i].type), blocks[i].topology, exoid);
        ex_err_fn(exoid, __func__, errmsg, status);
        goto error_ret; /* exit define mode and return */
      }
    }

    if (arbitrary_polyhedra == 0) {
      if (vedgcon && blocks[i].num_edges_per_entry) {
        dims[0] = numblkdim;
        dims[1] = nedgperentdim;

        if ((status = nc_def_var(exoid, vedgcon, conn_int_type, 2, dims, &varid)) != EX_NOERR) {
          snprintf(errmsg, MAX_ERR_LENGTH,
                   "ERROR: failed to create edge connectivity array for %s %" PRId64
                   " in file id %d",
                   ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
          ex_err_fn(exoid, __func__, errmsg, status);
          goto error_ret; /* exit define mode and return */
        }
      }

      if (vfaccon && blocks[i].num_faces_per_entry) {
        dims[0] = numblkdim;
        dims[1] = nfacperentdim;

        if ((status = nc_def_var(exoid, vfaccon, conn_int_type, 2, dims, &varid)) != EX_NOERR) {
          snprintf(errmsg, MAX_ERR_LENGTH,
                   "ERROR: failed to create face connectivity array for %s %" PRId64
                   " in file id %d",
                   ex_name_of_object(blocks[i].type), blocks[i].id, exoid);
          ex_err_fn(exoid, __func__, errmsg, status);
          goto error_ret; /* exit define mode and return */
        }
      }
    }
  }

  free(blocks_to_define);

  /* leave define mode  */
  if ((status = exi_leavedef(exoid, __func__)) != EX_NOERR) {
    snprintf(errmsg, MAX_ERR_LENGTH, "ERROR: failed to exit define mode");
    ex_err_fn(exoid, __func__, errmsg, status);
    EX_FUNC_LEAVE(EX_FATAL);
  }

  for (i = 0; i < block_count; i++) {
    switch (blocks[i].type) {
    case EX_EDGE_BLOCK: vblkids = VAR_ID_ED_BLK; break;
    case EX_FACE_BLOCK: vblkids = VAR_ID_FA_BLK; break;
    case EX_ELEM_BLOCK: vblkids = VAR_ID_EL_BLK; break;
    default: EX_FUNC_LEAVE(EX_FATAL); /* should have been handled earlier; quiet compiler here */
    }

    nc_inq_varid(exoid, vblkids, &att_name_varid);

    if (blocks[i].num_attribute > 0 && att_name_varid >= 0) {
      /* Output a dummy empty attribute name in case client code doesn't
         write anything; avoids corruption in some cases.
      */
      size_t count[2];
      char  *text = "";

      count[0] = 1;
      start[1] = 0;
      count[1] = strlen(text) + 1;

      for (int64_t j = 0; j < blocks[i].num_attribute; j++) {
        start[0] = j;
        nc_put_vara_text(exoid, att_name_varid, start, count, text);
      }
    }
  }

  EX_FUNC_LEAVE(EX_NOERR);

/* Fatal error: exit definition mode and return */
error_ret:
  free(blocks_to_define);

  exi_leavedef(exoid, __func__);
  EX_FUNC_LEAVE(EX_FATAL);
}
